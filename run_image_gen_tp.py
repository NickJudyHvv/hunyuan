# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/Tencent-Hunyuan/HunyuanImage-3.0/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import torch
import torch_npu
from torch import nn
import torch.distributed as dist

import os
from pathlib import Path
from hunyuan_image_3.hunyuan import HunyuanImage3ForCausalMM, HunyuanImage3DecoderLayer
import time

def parse_args():
    parser = argparse.ArgumentParser("Commandline arguments for running HunyuanImage-3 locally")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt to run")
    parser.add_argument("--model-id", type=str, default="./HunyuanImage-3", help="Path to the model")
    parser.add_argument("--attn-impl", type=str, default="sdpa", choices=["sdpa", "flash_attention_2"],
                        help="Attention implementation. 'flash_attention_2' requires flash attention to be installed.")
    parser.add_argument("--moe-impl", type=str, default="eager", choices=["eager", "flashinfer"],
                        help="MoE implementation. 'flashinfer' requires FlashInfer to be installed.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed. Use None for random seed.")
    parser.add_argument("--diff-infer-steps", type=int, default=50, help="Number of inference steps.")
    parser.add_argument("--image-size", type=str, default="auto",
                        help="'auto' means image size is determined by the model. Alternatively, it can be in the "
                             "format of 'HxW' or 'H:W', which will be aligned to the set of preset sizes.")
    parser.add_argument("--use-system-prompt", type=str,
                        choices=["None", "dynamic", "en_vanilla", "en_recaption", "en_think_recaption", "custom"],
                        help="Use system prompt. 'None' means no system prompt; 'dynamic' means the system prompt is "
                             "determined by --bot-task; 'en_vanilla', 'en_recaption', 'en_think_recaption' are "
                             "three predefined system prompts; 'custom' means using the custom system prompt. When "
                             "using 'custom', --system-prompt must be provided. Default to load from the model "
                             "generation config.")
    parser.add_argument("--system-prompt", type=str, help="Custom system prompt. Used when --use-system-prompt "
                                                          "is 'custom'.")
    parser.add_argument("--bot-task", type=str, choices=["image", "auto", "think", "recaption"],
                        help="Type of task for the model. 'image' for direct image generation; 'auto' for text "
                             "generation; 'think' for think->re-write->image; 'recaption' for re-write->image."
                             "Default to load from the model generation config.")
    parser.add_argument("--save", type=str, default="image.png", help="Path to save the generated image")
    parser.add_argument("--verbose", type=int, default=0, help="Verbose level")
    parser.add_argument("--rewrite", type=int, default=0, help="Whether to rewrite the prompt with DeepSeek")
    parser.add_argument("--sys-deepseek-prompt", type=str, choices=["universal", "text_rendering"], 
                        default="universal", help="System prompt for rewriting the prompt")

    parser.add_argument("--reproduce", action="store_true", help="Whether to reproduce the results")
    return parser.parse_args()


def set_reproducibility(enable, global_seed=None, benchmark=None):
    import torch
    if enable:
        # Configure the seed for reproducibility
        import random
        random.seed(global_seed)
        # Seed the RNG for Numpy
        import numpy as np
        np.random.seed(global_seed)
        # Seed the RNG for all devices (both CPU and CUDA)
        torch.manual_seed(global_seed)
    # Set following debug environment variable
    # See the link for details: https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
    if enable:
        import os
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # Cudnn benchmarking
    torch.backends.cudnn.benchmark = (not enable) if benchmark is None else benchmark
    # Use deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = enable
    torch.use_deterministic_algorithms(enable)

class ColumnParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, gather_output=True, tp_size=None, tp_rank=None, tp_group=None):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        super().__init__(in_features, out_features, bias)
    
    def forward(self, x):
        x = super().forward(x)
        return x

class RowParallelLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, input_is_parallel=True, tp_size=None, tp_rank=None, tp_group=None, matmul_allreduce_type="torch_npu"):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.tp_group = tp_group
        self.input_is_parallel = input_is_parallel
        self.matmul_allreduce_type = matmul_allreduce_type
        super().__init__(in_features, out_features, bias)
    
    def forward(self, x):
        if not self.input_is_parallel:
            x = torch.chunk(x, self.tp_size, dim=-1)[self.tp_rank]
        
        # x, b, s, h1; w h1, h2
        if self.matmul_allreduce_type == "torch_npu":
            hcom = self.tp_group._get_backend(torch.device("npu")).get_hccl_comm_name(self.tp_rank)
            output = torch_npu.npu_mm_all_reduce_base(x, self.weight.T, hcom)
        else:
            x = super().forward(x)
            # conduct all-reduce
            dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tp_group)
            # self.tp_group.allreduce(x)
            output = x
        return output

class TensorParallelApplicator:
    def __init__(self, tp_size, tp_rank, device_map="cpu", tp_group=None):
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.device_map = device_map
        self.tp_group = tp_group
    
    def apply_to_llm_model(self, model):
        self._replace_transformer_block(model)

    def _replace_transformer_block(self, module):
        for name, child in module.named_children():
            if isinstance(child, HunyuanImage3DecoderLayer):
                tp_size = self.tp_size
                tp_rank = self.tp_rank
                # self._replace_llm_attention(child.self_attn, tp_rank, tp_size)

                self._replace_llm_ffn(child.mlp, tp_rank, tp_size)
            else:
                self._replace_transformer_block(child)

    def _replace_llm_attention(self, child, tp_rank, tp_size):
        orig_wqkv = child.qkv_proj
        orig_wo = child.o_proj

        column_out = orig_wqkv.out_features // tp_size
        row_in = orig_wo.in_features // tp_size

        child.qkv_proj = ColumnParallelLinear(
            in_features=orig_wqkv.in_features,
            out_features=column_out,
            bias=False,
            gather_output=False,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=self.tp_group
        ).to(orig_wqkv.weight.dtype).to(self.device_map)

        child.o_proj = RowParallelLinear(
            in_features=row_in,
            out_features=orig_wo.out_features,
            bias=False,
            input_is_parallel=True,
            tp_size=tp_size,
            tp_rank=tp_rank,
            tp_group=self.tp_group
        ).to(orig_wo.weight.dtype).to(self.device_map)

        with torch.no_grad():
            q, k, v = torch.split(
                orig_wqkv.weight.data,
                [
                    child.num_key_value_groups * orig_wqkv.weight.data.shape[0] // (child.num_key_value_groups + 2),
                    1 * orig_wqkv.weight.data.shape[0] // (child.num_key_value_groups + 2),
                    1 * orig_wqkv.weight.data.shape[0] // (child.num_key_value_groups + 2)
                ],
                dim=0
            )
            q_chunk = torch.chunk(q.reshape(-1, orig_wqkv.weight.data.shape[1]), tp_size, dim=0)[tp_rank]
            k_chunk = torch.chunk(k.reshape(-1, orig_wqkv.weight.data.shape[1]), tp_size, dim=0)[tp_rank]
            v_chunk = torch.chunk(v.reshape(-1, orig_wqkv.weight.data.shape[1]), tp_size, dim=0)[tp_rank]
            wqkv_chunk = torch.cat([q_chunk, k_chunk, v_chunk])

            child.qkv_proj.weight.data = wqkv_chunk.contiguous()

            wo_chunk = torch.chunk(orig_wo.weight, tp_size, dim=1)[tp_rank]
            child.o_proj.weight.data = wo_chunk.contiguous()
        
        child.num_heads = child.num_heads // int(os.getenv("WORLD_SIZE", 8))
        child.num_key_value_heads = child.num_key_value_heads // int(os.getenv("WORLD_SIZE", 8))

    def _replace_llm_ffn(self, ff_layer, tp_rank, tp_size):
        for i in range(len(ff_layer.experts)):
            orig_gate_and_up_proj = ff_layer.experts[i].gate_and_up_proj
            orig_down_proj = ff_layer.experts[i].down_proj

            column_out = orig_gate_and_up_proj.out_features // tp_size
            row_in = orig_down_proj.in_features // tp_size

            ff_layer.experts[i].gate_and_up_proj = ColumnParallelLinear(
                in_features=orig_gate_and_up_proj.in_features,
                out_features=column_out,
                bias=False,
                gather_output=False,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=self.tp_group
            ).to(orig_gate_and_up_proj.weight.dtype).to(self.device_map)

            ff_layer.experts[i].down_proj = RowParallelLinear(
                in_features=row_in,
                out_features=orig_down_proj.out_features,
                bias=False,
                input_is_parallel=True,
                tp_size=tp_size,
                tp_rank=tp_rank,
                tp_group=self.tp_group
            ).to(orig_down_proj.weight.dtype).to(self.device_map)

            with torch.no_grad():
                gate_and_up_chunk = torch.cat(torch.chunk(orig_gate_and_up_proj.weight.data, tp_size * 2, dim=0)[tp_rank::tp_size])
                ff_layer.experts[i].gate_and_up_proj.weight.data = gate_and_up_chunk.contiguous()

                down_chunk = torch.chunk(orig_down_proj.weight, tp_size, dim=1)[tp_rank]
                ff_layer.experts[i].down_proj.weight.data = down_chunk.contiguous()

def main(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 8))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    stream = torch.npu.Stream()
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="hccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    ranks = list(range(world_size))
    tp_group = dist.new_group(ranks=ranks, backend="hccl")

    if args.reproduce:
        set_reproducibility(args.reproduce, global_seed=args.seed)

    if not args.prompt:
        raise ValueError("Prompt is required")
    if not Path(args.model_id).exists():
        raise ValueError(f"Model path {args.model_id} does not exist")

    kwargs = dict(
        attn_implementation=args.attn_impl,
        torch_dtype="auto",
        device_map="cpu",
        moe_impl=args.moe_impl,
    )
    model = HunyuanImage3ForCausalMM.from_pretrained(args.model_id, **kwargs)

    # TP
    tp_applicator = TensorParallelApplicator(
        tp_size=world_size,
        tp_rank=local_rank,
        device_map='cpu',
        tp_group=tp_group
    )
    tp_applicator.apply_to_llm_model(model.model)

    model.load_tokenizer(args.model_id)
    model.to(device)

    # Rewrite prompt with DeepSeek
    if args.rewrite:
        from PE.deepseek import DeepSeekClient
        from PE.system_prompt import system_prompt_universal, system_prompt_text_rendering

        # Get request key_id and key_secret for DeepSeek
        deepseek_key_id = os.getenv("DEEPSEEK_KEY_ID")
        deepseek_key_secret = os.getenv("DEEPSEEK_KEY_SECRET")
        if not deepseek_key_id or not deepseek_key_secret:
            raise ValueError(f"DeepSeek API key is not set!!! The Pretrain Checkpoint does not "
                             f"automatically rewrite or enhance input prompts, for optimal results currently,"
                             f"we recommend community partners to use deepseek to rewrite the prompts.")
        deepseek_client = DeepSeekClient(deepseek_key_id, deepseek_key_secret)
        
        if args.sys_deepseek_prompt == "universal":
            system_prompt = system_prompt_universal
        elif args.sys_deepseek_prompt == "text_rendering":
            system_prompt = system_prompt_text_rendering
        else:
            raise ValueError(f"Invalid system prompt: {args.sys_deepseek_prompt}")
        prompt, _ = deepseek_client.run_single_recaption(system_prompt, args.prompt)
        print("rewrite prompt: {}".format(prompt))
        args.prompt = prompt

    image = model.generate_image(
        prompt=args.prompt,
        seed=args.seed,
        image_size=args.image_size,
        use_system_prompt=args.use_system_prompt,
        system_prompt=args.system_prompt,
        bot_task=args.bot_task,
        diff_infer_steps=args.diff_infer_steps,
        verbose=args.verbose,
        stream=True,
    )
   
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)

    if local_rank == 0:
        image.save(args.save)
        print(f"Image saved to {args.save}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
