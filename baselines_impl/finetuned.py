import argparse
import json
import os
import sys
from pathlib import Path
from typing import List

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from configs.configs import DefaultTrainingArguments, peft_config
from configs.lang_codes import LangCodes
from modules.data import get_dataset, read_parallel_data
from utils.common_utils import free_gpu, set_special_tokens

lang_codes = LangCodes()
LANGS = {
    "zho_Hans": "Chinese",
    "eng_Latn": "English",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "hin_Deva": "Hindi",
    "spa_Latn": "Spanish",
    "tha_Thai": "Thai",
}
MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
]
# SFT_DATA_PATH = "dataset/flores200_dataset/sample_5k/flores200.parquet"
SFT_DATA_PATH = "dataset/flores200_dataset/sample_5k"
DATASET_DIR = "dataset/flores200_dataset/test/"
OUTPUT_DIR = "output/baseline/finetuned"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
PROMPT_TEMPLATE = "Please translate the {src_lang} into {trg_lang}: {src_sent}"
MAX_NEW_TOKENS = 200
GPU_UTILIZATION = 0.8

def get_args():
    parser = argparse.ArgumentParser(DefaultTrainingArguments)
    parser.add_argument("--llm_path", type=Path, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--padding_side", type=str, default="left")
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--truncation_side", type=str, default="left")

    args = parser.parse_args()
    return args


def convert_sft_dataset(sample):
    item = sample[0]
    valid_key = [k for k in item.keys() if item[k] is not None][:2]
    data = []
    for idx in range(2):
        if idx == 0:
            src_lang, trg_lang = valid_key
        else:
            trg_lang, src_lang = valid_key
        src_sent = item[src_lang]
        trg_sent = item[trg_lang]
        d = {
            "src_lang": src_lang,
            "trg_lang": trg_lang,
            "src_sent": src_sent,
            "trg_sent": trg_sent,
        }
        data.append(d)
    return data


def parse_sft_dataset(sample):
    src_lang = sample["src_lang"]
    trg_lang = sample["trg_lang"]
    src_sent = sample["src_sent"]
    trg_sent = sample["trg_sent"]

    src_lang = LANGS[src_lang] if src_lang in LANGS else lang_codes.get_lang(src_lang)
    trg_lang = LANGS[trg_lang] if trg_lang in LANGS else lang_codes.get_lang(trg_lang)

    input_sentence = PROMPT_TEMPLATE.format(
        src_lang=src_lang, trg_lang=trg_lang, src_sent=src_sent
    )
    input_context = {
        "messages": [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": input_sentence},
            {"role": "assistant", "content": trg_sent},
        ]
    }
    return input_context


def get_response_template(tokenizer):
    demo_message = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]
    origin = tokenizer.apply_chat_template(demo_message, tokenize=False)
    origin_plus_template = tokenizer.apply_chat_template(
        demo_message, tokenize=False, add_generation_prompt=True
    )
    generation_template = origin_plus_template.replace(origin, "")
    return generation_template


def sft_LLM(
    args: argparse.Namespace,
    data_path: Path = SFT_DATA_PATH,
):
    sft_dataset = get_dataset(data_path)
    converted_sft_dataset = sum([convert_sft_dataset(d) for d in sft_dataset], [])
    parsed_sft_dataset = [parse_sft_dataset(d) for d in converted_sft_dataset]
    train_dataset = Dataset.from_list(parsed_sft_dataset)

    tokenizer = AutoTokenizer.from_pretrained(
        args.llm_path,
        model_max_length=args.max_length,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        trust_remote_code=True,
    )
    tokenizer.padding_side = "right"
    response_template = get_response_template(tokenizer)
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template, tokenizer=tokenizer
    )

    llm = AutoModelForCausalLM.from_pretrained(args.llm_path, trust_remote_code=True)
    if args.use_lora:
        llm = get_peft_model(llm, peft_config=peft_config)
    llm.config.use_cache = False
    llm.is_parallelizable = True
    llm.model_parallel = True
    llm, tokenizer = set_special_tokens(llm, tokenizer)

    training_args = SFTConfig(
        max_seq_length=args.max_length,
        optim=args.optim,
        lr_scheduler_type=args.lr_scheduler_type,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        output_dir=args.output_dir,
        ddp_backend="nccl",
    )

    trainer = SFTTrainer(
        model=llm,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        data_collator=collator,
    )
    save_state_file = os.path.join(args.output_dir, "trainer_state.json")
    train_results = trainer.train(
        resume_from_checkpoint=True if os.path.exists(save_state_file) else None
    )
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if dist.get_rank() == 0:
        if args.use_lora:
            # cache the lora adaptor for debug
            trainer.save_model(output_dir=args.output_dir)
            llm = llm.merge_and_unload()
        llm.save_pretrained(args.output_dir, safe_serialization=True)
        tokenizer.save_pretrained(args.output_dir)

    trainer.accelerator.free_memory()
    del llm, tokenizer, trainer
    free_gpu()


def build_input(src_lang: str, trg_lang: str, tokenizer: AutoTokenizer):
    test_fpath = os.path.join(DATASET_DIR, f"flores_test_{src_lang}-{trg_lang}.parquet")
    src_list, trg_list = read_parallel_data(test_fpath, src_lang, trg_lang)
    input_sentences = [
        PROMPT_TEMPLATE.format(
            src_lang=LANGS[src_lang],
            trg_lang=LANGS[trg_lang],
            src_sent=sentence,
        )
        for sentence in src_list
    ]
    input_contexts = [
        [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": sentence},
        ]
        for sentence in input_sentences
    ]
    input_messages = [
        tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        for chat in input_contexts
    ]
    return src_list, input_messages, trg_list


def save_results(
    sources: List[str],
    outputs: List[str],
    references: List[str],
    src_lang: str,
    trg_lang: str,
    model_name: str,
):
    output_dir = os.path.join(OUTPUT_DIR, model_name)
    os.makedirs(output_dir, exist_ok=True)
    output_fpath = os.path.join(output_dir, f"flores_test_{src_lang}-{trg_lang}.json")
    context = [
        {"source": source, "predicted": output, "reference": reference}
        for source, output, reference in zip(sources, outputs, references)
    ]
    with open(output_fpath, "w+", encoding="utf-8") as f:
        json.dump(context, f, ensure_ascii=False, indent=2)


def infer(model_path: Path, model_name: str):
    from vllm import LLM, SamplingParams

    SAMPLING_PARAMS = SamplingParams(n=1, temperature=0, max_tokens=MAX_NEW_TOKENS)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model_path,
        dtype=torch.bfloat16,
        tokenizer=model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=GPU_UTILIZATION,
    )
    for src_lang in LANGS.keys():
        for trg_lang in LANGS.keys():
            if src_lang == trg_lang:
                continue
            src_list, input_messages, references = build_input(
                src_lang, trg_lang, tokenizer
            )
            generation_out = llm.generate(input_messages, SAMPLING_PARAMS)
            outputs = [item.outputs[0].text for item in generation_out]
            save_results(src_list, outputs, references, src_lang, trg_lang, model_name)


def main(args: argparse.Namespace):
    model_path = args.llm_path
    model_name = model_path.name
    args.output_dir = os.path.join("saved_models/finetuned", model_name)
    os.makedirs(args.output_dir, exist_ok=True)
    sft_LLM(args)
    if dist.get_rank() == 0:
        infer(args.output_dir, model_name)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    dist.init_process_group(backend="nccl", init_method="env://")
    opts = get_args()
    main(opts)
