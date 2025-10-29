export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=2
export TOKENIZERS_PARALLELISM=false

export ASCEND_LAUNCH_BLOCKING=1

python   run_image_gen_tp.py \
         --model-id /data/weights/HunyuanImage-3.0 \
         --verbose 1 \
         --sys-deepseek-prompt "universal" \
         --prompt "A brown and white dog is running on the grass" \
         --image-size 512x512 \
         --diff-infer-steps 50 \
         --seed 1234 \
         --reproduce