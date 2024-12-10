WANDB_PROJECT="zouw_debug" WANDB_NAME="llama3-debug" torchrun --master_addr $METIS_WORKER_0_HOST --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT --node_rank $ARNOLD_ID --nnodes $ARNOLD_WORKER_NUM main.py \
    --mode RL  --mcts_sample_size 10 \
    --llm_path models/huggingface/Llama-3.1-8b/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --src_code deu_Latn --trg_code arb_Arab
    --dev_data_path dataset/flores200_dataset/test/flores_test_deu_Latn-arb_Arab.parquet \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/llama3-debug/ \
    --flores_script "flores200.py" \
    --output_dir /mnt/bn/v2024/ckpts/llama3-debug/ \
    --deepspeed configs/ds_z2_config.json  --use_lora False \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-3 \
    --rl_learning_rate 3e-6 \
    --report_to 'wandb' \
    --run_name 'llama3-debug' \
    --bf16 True --tf32 True  2>&1 |tee contine.log

WANDB_PROJECT="zouw_debug" WANDB_NAME="llama3.2_debug" torchrun --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT  main.py \
    --mode RL  --mcts_sample_size 10 \
    --llm_path models/huggingface/Llama-3.2-3b/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --dev_data_path dataset/flores200_dataset/test/flores_test_eng_Latn-arb_Arab.parquet \
    --src_code eng_Latn --trg_code zho_Hans \
    --flores_script "flores200.py" \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/llama3.2_debug/ \
    --output_dir /mnt/bn/v2024/ckpts/llama3.2_debug/ \
    --deepspeed configs/ds_z2_config.json  --use_lora False \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-3 \
    --rl_learning_rate 2e-6 \
    --run_name 'llama3.2_debug' \
    --report_to 'wandb' \
    --bf16 True --tf32 True

CUDA_VISIBLE_DEVICES="0,1,2,3" python3 main.py \
    --mode valid \
    --llm_path models/huggingface/llama3.2_3b/ \
    --output_dir /mnt/bn/v2024/ckpts/llama3.2_debug/ \
    --cache_dir cache/llama3.2_debug/ --flores_script "flores200.py"  \
    --deepspeed configs/ds_z2_config.json \
    --nas_base_path  /mnt/bn/v2024/ \
    --per_device_eval_batch_size 4 \
    --tf32 True --bf16 True

CUDA_VISIBLE_DEVICES="0,1" python3 main.py \
    --mode air  --llm_path /mnt/bn/v2024/models/huggingface/Llama-3.1-8b \
    --output_dir /mnt/bn/v2024/ckpts/trans0_debug/ \
    --bf16 True --tf32 True

CUDA_VISIBLE_DEVICES="0,1" python3 main.py \
    --mode air  --output_dir /mnt/bn/v2024/models/huggingface/mistral_v2_ct_31.5w_sft_3e-6/ \
    --bf16 True --tf32 True --use_lora=False

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port=8008 main.py \
    --mode simulate --mcts_sample_size 10 \
    --output_dir /mnt/bn/v2024/ckpts/llama3-mega_sft1e-3/_RL  \
    --nas_base_path /mnt/bn/v2024/ \
    --cache_dir cache/debug/ \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --use_lora False \
    --bf16 True --tf32 True 2>&1 |tee mc_tree.log