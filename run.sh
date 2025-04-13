WANDB_PROJECT="trans0" WANDB_NAME="llama3.1_deu2zho" torchrun --master_addr $METIS_WORKER_0_HOST --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT --node_rank $ARNOLD_ID --nnodes $ARNOLD_WORKER_NUM main.py \
    --mode RL  --mcts_sample_size 5 \
    --llm_path models/huggingface/Llama-3.1-8b/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/llama3.1_trans0/ \
    --flores_script "flores200.py" \
    --src_code deu_Latn --trg_code zho_Hans  \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0/ \
    --deepspeed configs/ds_z2_config.json \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-4 \
    --rl_learning_rate 1e-6 \
    --report_to 'wandb' \
    --run_name 'llama3.1_deu2zho' \
    --bf16 True --tf32 True  2>&1 |tee contine.log

WANDB_PROJECT="debug" WANDB_NAME="llama3.1_debug" torchrun --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT  main.py \
    --mode RL  --mcts_sample_size 5 \
    --llm_path models/huggingface/Llama-3.1-8b/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --src_code deu_Latn --trg_code zho_Hans \
    --flores_script "flores200.py" \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/llama3.1_debug/ \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_debug/ \
    --deepspeed configs/ds_z2_config.json \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-4 \
    --rl_learning_rate 1e-6 \
    --run_name 'llama3.1_debug' \
    --report_to 'wandb' \
    --bf16 True --tf32 True 2>&1 |tee contine.log

WANDB_PROJECT="debug" WANDB_NAME="qwen_baseline" torchrun --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT  main.py \
    --mode SFT  \
    --llm_path models/huggingface/Qwen2.5-7B/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/qwen_debug/ \
    --output_dir /mnt/bn/v2024/ckpts/qwen_debug/ \
    --deepspeed configs/ds_z2_config.json \
    --learning_rate 3e-6 \
    --run_name 'qwen_baseline' \
    --report_to 'wandb' \
    --bf16 True --tf32 True 2>&1 |tee contine.log

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 main.py \
    --mode valid \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --cache_dir cache/llama3.1_trans0/ --flores_script "flores200.py"  \
    --deepspeed configs/ds_z2_config.json \
    --nas_base_path  /mnt/bn/v2024/ \
    --tf32 True --bf16 True  --valid_type "en2x"

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 main.py \
    --mode test \
    --test_data_path sample.txt \
    --src_code "zho_Hans" --trg_code "eng_Latn" \
    --output_dir /mnt/bn/v2024/ckpts/llama3.2_debug/ \
    --cache_dir cache/llama3.2_debug/  \
    --deepspeed configs/ds_z2_config.json \
    --nas_base_path  /mnt/bn/v2024/ \
    --tf32 True --bf16 True

CUDA_VISIBLE_DEVICES="0,1" python3 main.py \
    --mode air \
    --output_dir /mnt/bn/v2024/ckpts/baselines/llama3.1_sft \
    --bf16 True --tf32 True

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port=8008 main.py \
    --mode simulate --mcts_sample_size 5 \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0  \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --nas_base_path /mnt/bn/v2024/ \
    --cache_dir cache/llama3.1_trans0/ \
    --bf16 True --tf32 True 2>&1 |tee mc_tree.log

torchrun --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT  main.py  \
    --mode valid++  --mcts_sample_size 5 \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --cache_dir cache/llama3.1_trans0/ --flores_script "flores200.py"  \
    --nas_base_path  /mnt/bn/v2024/ \
    --tf32 True --bf16 True --valid_type "en2x"