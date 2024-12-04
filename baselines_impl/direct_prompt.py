import argparse
import json
import os
import sys
from typing import List

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from modules.data import read_parallel_data

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
    "arb_Arab": "Arabic",
    "isl_Latn": "Icelandic",
}
MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
]
DATASET_DIR = "dataset/flores200_dataset/test/"
OUTPUT_DIR = "output/baseline/direct"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
PROMPT_TEMPLATE = "Please translate the {src_lang} into {trg_lang}: {src_sent}."
MAX_NEW_TOKENS = 200
GPU_UTILIZATION = 0.8
SAMPLING_PARAMS = SamplingParams(n=1, temperature=0, max_tokens=MAX_NEW_TOKENS)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=MODELS, default=None)
    return parser.parse_args()


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


def infer(model_name: str):
    model_path = os.path.join("models/huggingface", model_name)
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
    model_name = args.model_name
    infer(model_name)


if __name__ == "__main__":
    opts = parse_args()
    main(opts)
