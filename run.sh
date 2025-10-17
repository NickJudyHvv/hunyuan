export ASCEND_RT_VISIBLE_DEVICES=0
export WORLD_SIZE=1
export ALGO=0
export PYTORCH_NPU_ALLOC_CONF='expand_segements:True'
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1
export TOKENIZERS_PARALLELISM=false

export ASCEND_LAUNCH_BLOCKING=1

torchrun --nproc_per_node=${WORLD_SIZE} run_image_gen_tp.py \
         --model-id /data/weights/HunyuanImage-3.0 \
         --verbose 1 \
         --sys-deepseek-prompt "universal" \
         --prompt "A brown and white dog is running on the grass" \
         --image-size 512x512 \
         --diff-infer-steps 50 \
         --seed 1234 \
         --reproduce