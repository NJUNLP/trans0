WANDB_PROJECT="zouw_debug" WANDB_NAME="llama3-debug" torchrun --master_addr $METIS_WORKER_0_HOST --nproc_per_node $NUM_GPU --master_port $WORKER_0_PORT --node_rank $ARNOLD_ID --nnodes $ARNOLD_WORKER_NUM main.py \
    --mode RL  --mcts_sample_size 5 \
    --llm_path models/huggingface/Llama-3.1-8b/  \
    --train_data_path dataset/flores200_dataset/sample_40k/ \
    --dev_data_path dataset/flores200_dataset/test/flores_test_zho_Hans-eng_Latn.parquet \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/llama3-debug/ \
    --flores_script "flores200.py" \
    --output_dir /mnt/bn/v2024/ckpts/llama3-debug/ \
    --deepspeed configs/ds_z2_config.json  --use_lora False \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-4 \
    --rl_learning_rate 5e-6 \
    --report_to 'wandb' \
    --run_name 'llama3-debug' \
    --bf16 True --tf32 True  2>&1 |tee contine.log

torchrun --nproc_per_node=8 --master_port=8009 main.py \
    --mode valid --src_code eng_Latn --trg_code zho_Hans \
    --output_dir /mnt/bn/v2024/ckpts/llama3-debug/ \
    --cache_dir cache/llama3-debug/ \
    --dev_data_path /mnt/bn/v2024/dataset/flores200_dataset/test/flores_test_zho_Hans-eng_Latn.parquet \
    --deepspeed configs/ds_z2_config.json \
    --nas_base_path  /mnt/bn/v2024/ \
    --per_device_eval_batch_size 4 \
    --bf16 True \
    --tf32 True

CUDA_VISIBLE_DEVICES="0,1" python3 main.py \
    --mode air  --llm_path /mnt/bn/v2024/models/huggingface/Llama-3.1-8b \
    --output_dir /mnt/bn/v2024/ckpts/trans0_debug/ \
    --bf16 True --tf32 True

CUDA_VISIBLE_DEVICES="0,1" python3 main.py \
    --mode air  --output_dir /mnt/bn/v2024/models/huggingface/mistral_v2_ct_31.5w_sft_3e-6/ \
    --bf16 True --tf32 True --use_lora=False

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port=8008 main.py \
    --mode simulate \
    --llm_path /mnt/bn/v2024/models/huggingface/Llama-3.1-8b/ \
    --output_dir /mnt/bn/v2024/ckpts/llama3-debug/ \
    --nas_base_path /mnt/bn/v2024/ \
    --cache_dir cache/debug/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --bf16 True --tf32 True 2>&1 |tee mc_tree.log