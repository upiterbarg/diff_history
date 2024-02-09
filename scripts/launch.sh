export CMAKE_CUDA_COMPILER=/usr/local/cuda-11.6/bin/nvcc
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NUM_GPUS=4
export BATCH_SIZE_PER_GPU=4
export TOTAL_BATCH_SIZE=256
export GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --main_process_port 29501 \
    --deepspeed_config_file ds_configs/stage3_offloading_accelerate.conf \
    finetune.py \
    --tokenizer_name gpt2 \
    --use_slow_tokenizer \
    --model_name_or_path models/gpt2_4096 \
    --seed 1 \
    --mask_observation_compl \
    --train_file ???  \
    --max_seq_length 4096 \
    --tracking_project ??? \
    --tracking_group ??? \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate 3e-4 \
    --checkpointing_steps 100 \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 32 \
    --with_tracking \
    --output_dir ??? \
    --report_to wandb \
    --logging_steps 1 
