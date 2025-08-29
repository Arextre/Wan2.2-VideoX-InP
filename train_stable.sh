export MODEL_NAME="/home/notebook/data/group/zhaoheng/pretrained_models/Wan2.2-Fun-5B-InP"
# export DATASET_NAME="datasets/internal_datasets/"
export DATASET_META_NAME="./metadata.json"
# NCCL_IB_DISABLE=1 and NCCL_P2P_DISABLE=1 are used in multi nodes without RDMA. 
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
NCCL_DEBUG=INFO
export NCCL_P2P_LEVEL=NVL

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
echo "Training started at: $(date)"
echo "Timestamp: $TIMESTAMP"

OUTPUT_DIR="./train_output/wan2.2/no_person_cityscene_adapter_full"
mkdir -p "$OUTPUT_DIR"
echo "Output save at location: $OUTPUT_DIR"

export NCCL_BUFFSIZE=2097152
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

/home/notebook/code/personal/S9060429/.venv/bin/accelerate launch --mixed_precision="bf16" train.py \
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
  --num_train_epochs=2 \
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
  --resume_from_checkpoint="latest" \
  --trainable_modules "."