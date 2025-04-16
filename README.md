
## Overview
Trans0 aims to initialize a multilingual LLM as a translation agent via monolingual data.
This is a public version with all in-house implementation replaced by huggingface trl.

scripts are in the run.sh

----
## Data preparation:
monolingual dataset are arranged as the following:
```
monolingual/
├── eng_Latn/
│   └── merge.txt
├── deu_Latn/
│   └── merge.txt
└── ....
```
Each path corresponds to the monolingual language code (ISO-2_code)
merge.txt are lines of monolingual sentence.

the instruct data for baselines and instruction tuning is extracted from the official flores200.py scripts.
the supervised data are arranged in parquet file.
check the collect_data.py to extract parallel data from the flores200.py

```
python3 collect_data.py  # go to the collect_data.py for specific data extraction lines.
```

----
## Launch scripts

parameters:
- --mode: SFT (baselines), RL (TransZero), test (test the data), valid (valid on specific ), valid++ (test with G-MCTS), simulate (run a simulated G-MCTS test)
- --nas_base_path: the base path setting, redundant if you are runing with full directories in scripts
- --cache_dir: path to the cached model if vLLM involved, also additional cache during self-plays and preference training
- --output_dir: the results including the trained LLM, valid and test results.
- --llm_path: the LLM path with its vocabulary
- --train_data_path: data lines for SFT training
- --learning_rate: SFT learning rate
- --self_play_languages: language codes for the self-play (ISO-2_code).
- --src_code and trg_code: languages codes for the training logs.
- --deepspeed the deepspeed config file.


Launching SFT baseline (remember to set the NUM_WORK_GPU and WORKER_*_PORT accordingly):
```
WANDB_PROJECT="debug" WANDB_NAME="qwen_baseline" torchrun --nproc_per_node $NUM_WORKER_GPU --master_port $WORKER_0_PORT  main.py \
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
```

Launching Trans0 (RL)
```
WANDB_PROJECT="debug" WANDB_NAME="llama3.1_debug" torchrun --nproc_per_node $NUM_WORKER_GPU --master_port $WORKER_0_PORT  main.py \
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
```


validation on all self_play languages (with vLLM):
- --valid_type: en2x, x2x, or x2en
```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 main.py \
    --mode valid \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --cache_dir cache/llama3.1_trans0/ --flores_script "flores200.py"  \
    --deepspeed configs/ds_z2_config.json \
    --nas_base_path  /mnt/bn/v2024/ \
    --tf32 True --bf16 True  --valid_type "en2x"
```

test
- --test_data_path: lines of translation test test data.
- --src_code and trg_code: indicates the translation src and trg
```
CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python3 main.py \
    --mode test \
    --test_data_path sample.txt \
    --src_code "zho_Hans" --trg_code "eng_Latn" \
    --output_dir /mnt/bn/v2024/ckpts/llama3.2_debug/ \
    --cache_dir cache/llama3.2_debug/  \
    --deepspeed configs/ds_z2_config.json \
    --nas_base_path  /mnt/bn/v2024/ \
    --tf32 True --bf16 True
```

launch G-MCTS on an input text
check the utils/unit_test.py for the unit_test() method for details.
```
CUDA_VISIBLE_DEVICES="0" torchrun --nproc_per_node=1 --master_port=8008 main.py \
    --mode simulate --mcts_sample_size 5 \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0  \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --nas_base_path /mnt/bn/v2024/ \
    --cache_dir cache/llama3.1_trans0/ \
    --bf16 True --tf32 True 2>&1 |tee mc_tree.log
```

valid++ (Note: test via G-MCTS is expensive and time-consuming, which is not recommended)
valid++ will extract the flores200 given all self_play_languages for specific validation.
```
torchrun --nproc_per_node $NUM_WORKER_GPU --master_port $WORKER_0_PORT  main.py  \
    --mode valid++  --mcts_sample_size 5 \
    --output_dir /mnt/bn/v2024/ckpts/llama3.1_trans0/ \
    --self_play_languages "deu_Latn" "por_Latn" "ita_Latn" "eng_Latn" "zho_Hans" "rus_Cyrl" \
    --cache_dir cache/llama3.1_trans0/ --flores_script "flores200.py"  \
    --nas_base_path  /mnt/bn/v2024/ \
    --tf32 True --bf16 True --valid_type "en2x"
```


## Acknowledgement
Trans0 was supported by ByteDance Research (ByteDance Inc).
Trans0 are also funded by the National Science Foundation of China (No. 62376116, 62176120), research project of Nanjing University-China Mobile Joint Institute, and the Fundamental Research Funds for the Central Universities (No. 2024300507)
