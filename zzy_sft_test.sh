#!/bin/bash

export ARNOLD_WORKER_NUM=1

NUM_GPU=4
MASTER_PORT=8099

WANDB_PROJECT="zzy_debug_trans0" WANDB_NAME="llama3-debug" torchrun \
    --nproc_per_node $NUM_GPU \
    --master_port $MASTER_PORT \
    main.py \
    --mode RL  --mcts_sample_size 10 \
    --llm_path models/huggingface/Llama-3.2-1B/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --dev_data_path dataset/flores200_dataset/test/flores_test_zho_Hans-eng_Latn.parquet \
    --nas_base_path ""  \
    --cache_dir cache/llama3-debug/ \
    --flores_script "flores200.py" \
    --output_dir ckpts/llama3-debug/ \
    --deepspeed configs/ds_z2_config.json  --use_lora False \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-3 \
    --rl_learning_rate 5e-6 \
    --report_to 'wandb' \
    --run_name 'llama3-debug' \
    --bf16 True --tf32 True  2>&1 |tee contine.log