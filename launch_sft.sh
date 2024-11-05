# script for Huawei Ascend910b
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
WANDB_PROJECT="zouw_debug" WANDB_NAME="llama2-sppo" torchrun --nproc_per_node=8 --master_port 8008 main.py \
    --mode RL  --mcts_sample_size 4 \
    --llm_path models/huggingface/Llama-2-7b-hf/ \
    --train_data_path dataset/flores200_dataset/sample_40k/ \
    --dev_data_path dataset/flores200_dataset/test/flores_test_zho_Hans-eng_Latn.parquet \
    --nas_base_path /mnt/bn/v2024/  \
    --cache_dir cache/trans0_llama2-7b \
    --flores_script "flores200.py" \
    --output_dir /mnt/bn/v2024/ckpts/trans0_llama2-7b/ \
    --rl_loss_type sppo_hard \
    --report_to 'wandb' \
    --run_name 'llama2-sppo' \
    --bf16 True --tf32 True  2>&1 |tee contine.log

    #nastk cp -a /mnt/bn/v2024/dataset bytenas://cn:v2024yg:<key>/