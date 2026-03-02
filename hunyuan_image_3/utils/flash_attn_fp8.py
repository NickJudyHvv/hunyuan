import math
import torch
import torch_npu

try:
    from msmodelslim.processor.quarot.common.quarot_utils import create_rot, QuaRotMode
    from mindiesd.layers.quant.block_quant import fa_block_quant_preprocess
except:
    print(f"msmodelslim or mindiesd quant is no enabled.\n")


class FP8RotateQuantFA(torch.nn.Module):
    def __init__(self, prefix=None, weights=None):
        super().__init__()

        rot_matrix = create_rot(QuaRotMode.HADAMARD, 128, seed=425500)
        self.register_buffer("rot_matrix", rot_matrix, persistent=False)

    def preprocess(self, query, key):
        if query.device != self.rot_matrix.device:
            self.rot_matrix = self.rot_matrix.to(query.device)
        query = torch.matmul(query, self.rot_matrix)
        key = torch.matmul(key, self.rot_matrix)
        return query, key

    def forward(self, query, key, value, **kwargs):
        layout = kwargs.get("layout", "BNSD")

        if 'enable_preprocess' not in kwargs or not kwargs['enable_preprocess']:
            query, key = self.preprocess(query, key)

        q, q_scale = fa_block_quant_preprocess(query, block_size=128,
                                               dst_type=torch_npu.float8_e4m3fn, layout=layout)
        k, k_scale = fa_block_quant_preprocess(key, block_size=256,
                                               dst_type=torch_npu.float8_e4m3fn, layout=layout)
        v, v_scale = fa_block_quant_preprocess(value, block_size=256,
                                               dst_type=torch_npu.float8_e4m3fn, layout=layout)

        if layout == "BNSD":
            _, n, s, d = query.shape
            _, kv_n, _, _ = key.shape
        elif layout == "BSND":
            _, s, n, d = query.shape
            _, _, kv_n, _ = key.shape

        x = torch_npu.npu_fused_infer_attention_score_v2(q, k, v, input_layout=layout,
                                                            num_query_heads=n,
                                                            num_key_value_heads=kv_n,
                                                            softmax_scale=1.0 / math.sqrt(d),
                                                            pre_tokens=2147483647,
                                                            next_tokens=2147483647,
                                                            query_quant_mode=7,
                                                            key_quant_mode=7,
                                                            value_quant_mode=7,
                                                            dequant_scale_query=q_scale,
                                                            dequant_scale_key=k_scale,
                                                            dequant_scale_value=v_scale,
                                                            out_dtype=query.dtype
                                                            )[0]

        if x.shape[2] != s:
            if layout == "BNSD":
                x = x[:, :, :s, :]
            elif layout == "BSND":
                x = x[:, :s, :, :]

        return x
