import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.distributed as dist
from datasets import load_dataset
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase
from trl import SFTConfig, SFTTrainer
from trl.trainer.utils import pad
from vllm import LLM, SamplingParams

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from configs.configs import peft_config
from configs.lang_codes import LangCodes
from modules.data import get_dataset, read_parallel_data, sft_data_collactor
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
SAMPLING_PARAMS = SamplingParams(n=1, temperature=0, max_tokens=MAX_NEW_TOKENS)


@dataclass
class DataCollatorForChatML:
    """
    Data collator for ChatML format datasets.
    Copied and Fixed from TRL source code trl/trainer/utils.py
    """

    tokenizer: PreTrainedTokenizerBase
    ignore_index: int = -100
    max_length: int = None
    prompt_key: str = "prompt"
    messages_key: str = "messages"

    def __post_init__(self):
        if self.tokenizer.pad_token_id is None:
            raise ValueError(
                "The tokenizer does not have a pad token. Please set `pad_token_id` in the tokenizer."
            )
        if self.max_length is None:
            # set a sensible default
            self.max_length = min(self.tokenizer.model_max_length, 1024)

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = []
        attention_mask = []
        prompts_input_ids = []
        prompt_attention_mask = []
        labels = []

        for example in examples:
            formatted_prompt = example.get(self.prompt_key, None)
            if formatted_prompt is None:
                prompt = example[self.messages_key][:-1]
                formatted_prompt = self.tokenizer.apply_chat_template(
                    prompt, tokenize=False, add_generation_prompt=True
                )

            if "input_ids" not in example:
                message = example[self.messages_key]
                formatted_message = self.tokenizer.apply_chat_template(
                    message, tokenize=False, add_generation_prompt=False
                )
                tokenized_message = self.tokenizer(
                    formatted_message,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                    return_tensors=None,
                    add_special_tokens=False,
                )
                input_ids.append(tokenized_message["input_ids"])
                attention_mask.append(tokenized_message["attention_mask"])
            else:
                input_ids.append(example["input_ids"])
                attention_mask.append(example["attention_mask"])

            tokenized_prompt = self.tokenizer(
                formatted_prompt,
                truncation=True,
                max_length=len(input_ids[-1]),
                padding=False,
                return_tensors=None,
                add_special_tokens=False,
            )

            prompts_input_ids.append(tokenized_prompt["input_ids"])
            prompt_attention_mask.append(tokenized_prompt["attention_mask"])

            # Create the labels that will have all but the completion tokens of the example["input_ids"] set to ignore_index
            label = [self.ignore_index] * len(input_ids[-1])
            completion_start_idx = len(tokenized_prompt["input_ids"])
            label[completion_start_idx:] = input_ids[-1][completion_start_idx:]
            labels.append(label)

        # convert to list of tensors and pad
        input_ids = [torch.tensor(ids, dtype=torch.long) for ids in input_ids]
        attention_mask = [
            torch.tensor(mask, dtype=torch.long) for mask in attention_mask
        ]
        labels = [torch.tensor(label, dtype=torch.long) for label in labels]
        input_ids = pad(
            input_ids, padding_side="left", padding_value=self.tokenizer.pad_token_id
        )
        attention_mask = pad(attention_mask, padding_side="left", padding_value=0)
        labels = pad(labels, padding_side="left", padding_value=self.ignore_index)

        prompts_input_ids = [
            torch.tensor(ids, dtype=torch.long) for ids in prompts_input_ids
        ]
        prompt_attention_mask = [
            torch.tensor(mask, dtype=torch.long) for mask in prompt_attention_mask
        ]
        prompts_input_ids = pad(
            prompts_input_ids,
            padding_side="left",
            padding_value=self.tokenizer.pad_token_id,
        )
        prompt_attention_mask = pad(
            prompt_attention_mask, padding_side="left", padding_value=0
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompts": prompts_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
        }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=MODELS, required=True)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--padding_side", type=str, default="left")
    parser.add_argument("--truncation_side", type=str, default="left")
    parser.add_argument("--use_lora", action="store_true")
    return parser.parse_args()


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


def sft_LLM(
    args: argparse.Namespace,
    model_path: Path,
    save_path: Path,
    data_path: Path = SFT_DATA_PATH,
):
    sft_dataset = get_dataset(data_path)
    converted_sft_dataset = sum([convert_sft_dataset(d) for d in sft_dataset], [])
    parsed_sft_dataset = [parse_sft_dataset(d) for d in converted_sft_dataset]

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=args.max_length,
        padding_side=args.padding_side,
        truncation_side=args.truncation_side,
        trust_remote_code=True,
    )

    llm = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    if args.use_lora:
        llm = get_peft_model(llm, peft_config=peft_config)
    llm.config.use_cache = False
    llm.is_parallelizable = True
    llm.model_parallel = True
    llm, tokenizer = set_special_tokens(llm, tokenizer)

    collator = DataCollatorForChatML(
        tokenizer,
        max_length=args.max_length,
        messages_key="messages",
        ignore_index=-100,
    )

    training_args = SFTConfig(
        max_seq_length=args.max_length,
        output_dir=save_path,
    )

    trainer = SFTTrainer(
        model=llm,
        args=training_args,
        train_dataset=parsed_sft_dataset,
        collator=collator,
        processing_class=tokenizer,
    )
    save_state_file = os.path.join(save_path, "trainer_state.json")
    train_results = trainer.train(
        resume_from_checkpoint=True if os.path.exists(save_state_file) else None
    )
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if dist.get_rank() == 0:
        if args.use_lora:
            trainer.save_model(output_dir=save_path)  # cache the lora adaptor for debug
            llm = llm.merge_and_unload()
        llm.save_pretrained(save_path, safe_serialization=True)
        tokenizer.save_pretrained(save_path)

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
    model_path = os.path.join("models/huggingface", model_name)
    model_path = "/Users/nil/Downloads/Llama-3.2-1B-Instruct"
    # model_path = "/Users/nil/Downloads/Qwen2.5-0.5B-Instruct"
    save_path = os.path.join("saved_models/baseline", model_name)
    sft_LLM(args, model_path, save_path)
    infer(save_path, model_name)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    opts = get_args()
    main(opts)
