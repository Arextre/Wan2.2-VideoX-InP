export MODEL_NAME="/home/notebook/data/group/zhaoheng/pretrained_models/Wan2.2-Fun-5B-InP"
# export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="./metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO

# 多GPU内存优化设置
export NCCL_BUFFSIZE=2097152  # 减少通信缓冲区大小
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128  # 减少内存碎片

if [ -z "$1" ]; then
  echo "Usage: $0 <GPUS>"
  echo "Example: $0 1  # for single GPU"
  echo "Example: $0 4  # for 4 GPUs"
  exit 1
fi
GPUS=$1

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Training started at: $(date)"
echo "Timestamp: $TIMESTAMP"

OUTPUT_DIR="./train_output/wan2.2/no_person_cityscene_adapter_full_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

ACCELERATE_CONF="$OUTPUT_DIR/accelerate.yaml"
if [ "$GPUS" -eq 1 ]; then
  DISTRIBUTED_TYPE="NO"
else
  DISTRIBUTED_TYPE="MULTI_GPU"
fi
# DISTRIBUTED_TYPE="DEEPSPEED"
cat > $ACCELERATE_CONF << EOF
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  gradient_accumulation_steps: 1
  zero3_init_flag: false
  zero_stage: 2
  train_micro_batch_size_per_gpu: 1
  fp16:
    enabled: true
    auto_cast: true
distributed_type: $DISTRIBUTED_TYPE
downcast_bf16: 'no'
enable_cpu_affinity: false
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: $GPUS
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
EOF

/home/notebook/code/personal/S9060429/.venv/bin/accelerate launch --config_file ./$ACCELERATE_CONF --mixed_precision="bf16" train.py \
  --config_path="wan_civitai_5b.yaml" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_meta=$DATASET_META_NAME \
  --image_sample_size=512 \
  --video_sample_size=512 \
  --token_sample_size=768 \
  --video_sample_stride=3 \
  --video_sample_n_frames=49 \
  --train_batch_size=1 \
  --video_repeat=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=4 \
  --num_train_epochs=20 \
  --checkpointing_steps=500 \
  --learning_rate=1e-5 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps=100 \
  --seed=42 \
  --output_dir=$OUTPUT_DIR \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --adam_weight_decay=5e-3 \
  --adam_epsilon=1e-10 \
  --vae_mini_batch=1 \
  --max_grad_norm=1.0 \
  --random_hw_adapt \
  --training_with_video_token_length \
  --enable_bucket \
  --uniform_sampling \
  --low_vram \
  --boundary_type="low" \
  --train_mode="inpaint" \
  --use_deepspeed \
  --trainable_modules "."