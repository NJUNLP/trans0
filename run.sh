WANDB_PROJECT="zouw_trans0" WANDB_NAME="llama3.1_deu2arb" torchrun --master_addr $METIS_WORKER_0_HOST --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT --node_rank $ARNOLD_ID --nnodes $ARNOLD_WORKER_NUM main.py \
    --mode RL  --mcts_sample_size 5 \
    --llm_path models/huggingface/Llama-3.1-8b/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "hin_Deva" "zho_Hans" "arb_Arab" \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/llama3.1_trans0/ \
    --flores_script "flores200.py" \
    --src_code deu_Latn --trg_code arb_Arab  \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0/ \
    --deepspeed configs/ds_z2_config.json \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-3 \
    --rl_learning_rate 5e-6 \
    --report_to 'wandb' \
    --run_name 'llama3.1_deu2arb' \
    --bf16 True --tf32 True  2>&1 |tee contine.log

WANDB_PROJECT="zouw_debug" WANDB_NAME="llama3.2_debug" torchrun --nproc_per_node $ARNOLD_WORKER_GPU --master_port $METIS_WORKER_0_PORT  main.py \
    --mode RL  --mcts_sample_size 5 \
    --llm_path models/huggingface/Llama-3.2-3b/  \
    --train_data_path dataset/flores200_dataset/sample_5k/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "hin_Deva" "zho_Hans" "arb_Arab" \
    --src_code deu_Latn --trg_code arb_Arab \
    --flores_script "flores200.py" \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/llama3.2_debug/ \
    --output_dir /mnt/bn/v2024/ckpts/llama3.2_debug/ \
    --deepspeed configs/ds_z2_config.json \
    --rl_loss_type sppo_hard \
    --learning_rate 1e-3 \
    --rl_learning_rate 5e-6 \
    --run_name 'llama3.2_debug' \
    --report_to 'wandb' \
    --bf16 True --tf32 True 2>&1 |tee contine.log

CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 main.py \
    --mode valid \
    --output_dir /mnt/bn/v2024/ckpts/clax_mega/_RL_2e-6 \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "hin_Deva" "zho_Hans" "arb_Arab" "rus_Cyrl"  \
    --cache_dir cache/clax-mega/ --flores_script "flores200.py"  \
    --deepspeed configs/ds_z2_config.json \
    --nas_base_path  /mnt/bn/v2024/ \
    --tf32 True --bf16 True

CUDA_VISIBLE_DEVICES="0,1" python3 main.py \
    --mode air  --llm_path /mnt/bn/v2024/models/huggingface/Llama-3.1-8b \
    --output_dir /mnt/bn/v2024/ckpts/trans0_debug/ \
    --bf16 True --tf32 True

CUDA_VISIBLE_DEVICES="0,1" python3 main.py \
    --mode air  --output_dir /mnt/bn/v2024/models/huggingface/mistral_v2_ct_31.5w_sft_3e-6/ \
    --bf16 True --tf32 True --use_lora=False

CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port=8008 main.py \
    --mode simulate --mcts_sample_size 5 \
    --output_dir /mnt/bn/v2024/ckpts/llama3-mega_sft1e-3/_RL  \
    --nas_base_path /mnt/bn/v2024/ \
    --cache_dir cache/debug/ \
    --bf16 True --tf32 True 2>&1 |tee mc_tree.log