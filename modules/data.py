# -*- coding: utf-8 -*-
import json, os, random
import torch
from typing import List
import pandas as pd
from pandas import DataFrame
import numpy as np

from copy import deepcopy
from utils.common_utils import print_once
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer
from configs.prompts import TRANS_PROMPT, LABEL_MARK
from configs.lang_codes import LangCodes

from datasets import load_dataset
IGNORE_INDEX=-100

SEP_TOKEN="<sep>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

lang_codes = LangCodes()

class TextDataset(Dataset):
    def __init__(self, data):
        self.data = data 

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self,):
        return len(self.data)

def read_json_or_jsonl_data(data_path):
    print_once(f">>> load data from {data_path}")
    if data_path.endswith(".json"):  # direct loading json files 
        with open(data_path, 'r') as f:
            data_list = json.load(f)
    elif data_path.endswith(".jsonl"):
        with open(data_path, 'r') as f:  # jsonl raw data (line as sample)
            lines = f.read().strip().split('\n')
            data_list = [json.loads(l) for l in lines]
    elif data_path.endswith(".parquet"):
        df = pd.read_parquet(data_path)  # wmt parquet is organized with key "translation"
        data_list = df.values.tolist()  # loading parquet data as list
    else:
        with open(data_path, "r") as f:  # raw line as sample
            data_list = [l.strip() for l in f]
    return data_list

def read_parallel_data(data_path, src_lang_code, trg_lang_code):
    """
    extract parallel data columns by language codes as lists
    """
    assert data_path.endswith(".parquet"), "must be a parquet file"
    print_once(f">>>load data from {data_path}")
    df = pd.read_parquet(data_path)
    src_list = df[df.keys().item()].apply(lambda x: x.get(src_lang_code, None)).tolist()
    trg_list = df[df.keys().item()].apply(lambda x: x.get(trg_lang_code, None)).tolist()
    return src_list, trg_list


def get_dataset(train_data_path:str, show_info=False):
    # collect all json data under the directory
    all_train_data = []
    for root, dir, files in os.walk(train_data_path):
        for file in files:
            if file.endswith(".json") or file.endswith(".jsonl") or file.endswith(".parquet"):
                train_data = read_json_or_jsonl_data(os.path.join(root,file))
                all_train_data.extend(train_data)

    if show_info:
        print_once(f">>> loaded data INFO:")        
        print_once(f">>> {all_train_data[0]}")

    train_set = TextDataset(all_train_data)
    return train_set

def batch_padding(input_ids, tokenizer, padding='longest', max_length=None, pad_token_id=None):
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    max_length = tokenizer.model_max_length if max_length is None else max_length
    
    if padding == 'longest':
        max_input_length = max([len(inp_ids) for inp_ids in input_ids])
        max_length = min(tokenizer.model_max_length, max_input_length)                

    outputs = {"input_ids": [], "attention_mask": []}
    for inp_ids in input_ids:        
        attn_mask = [1] * len(inp_ids)
        if len(inp_ids) >= max_length:
            if tokenizer.truncation_side == 'left':
                inp_ids = inp_ids[-max_length :]
                attn_mask = attn_mask[-max_length :]
            else:
                inp_ids = inp_ids[:max_length]
                attn_mask = attn_mask[:max_length]
        else:
            if tokenizer.padding_side == 'left':
                inp_ids = [pad_token_id] * (max_length - len(inp_ids)) + inp_ids
                attn_mask = [0] * (max_length - len(attn_mask)) + attn_mask
            else:
                inp_ids =  inp_ids + [pad_token_id] * (max_length - len(inp_ids)) 
                attn_mask = attn_mask + [0] * (max_length - len(attn_mask))

        outputs['input_ids'].append(deepcopy(inp_ids))
        outputs['attention_mask'].append(deepcopy(attn_mask))
    return outputs


def sft_data_collactor(batch, tokenizer:AutoTokenizer, show_info:bool):
    """
    yields the following:
    input_ids: long, the inputs, concat(instruct+input)
    attention_mask: float, indicating valid input ids, same size as input_ids
    labels: long, target outputs, for causalLM SFT, is the reference prepended with masked inputs 
      concat(IGNORE_index*(len(input_ids)+1) + outputs)

    add "for examples"
    """
    
    input_ids, attention_mask, labels, weights, rewards = [], [], [], [], []
    if show_info:
        print_once(f">>> batch INFO:")
        print_once(batch)

    for item in batch:  # sample level processing (tokenize), with random context samples as inputs.
        if "instruction" in item:  
            # alpaca data format with ["instruction", "input", "output"] region
            query = item["instruction"] + item["input"]+ LABEL_MARK
            output = item["output"]
        elif "context" in item:
            query = item["context"] + item["input"] + LABEL_MARK
            output =  item["output"]
        else:
            # raw language pairs from parquet files.
            item = item[0]
            valid_key = [k for k in item.keys() if item[k] is not None ][:2] # only first two language
            if random.uniform(0,1)>0.5:  # randomly choose to translate from src2trg or trg2src
                trg_lan_code, src_lan_code = valid_key
            else:
                src_lan_code, trg_lan_code = valid_key
            trans_prompt = random.choice(TRANS_PROMPT)
            instruction = trans_prompt.replace("<trg_lan>", lang_codes.get_lang(trg_lan_code))
            if "<src_lan>" in instruction:
                instruction = instruction.replace("<src_lan>", lang_codes.get_lang(src_lan_code))
            query = instruction.replace("<src_sent>", item[src_lan_code]) + LABEL_MARK
            output = item[trg_lan_code]

        query_token_ids = tokenizer.encode(query, add_special_tokens=False)
        target_token_ids = tokenizer.encode(output, add_special_tokens=False)
        input_ids.append(
            [tokenizer.bos_token_id] + deepcopy(query_token_ids) + deepcopy(target_token_ids) + [tokenizer.eos_token_id]
        )
        labels.append(
            [IGNORE_INDEX] * (len(query_token_ids)+1) + deepcopy(target_token_ids)+ [tokenizer.eos_token_id]
        )

    outputs = batch_padding(input_ids, tokenizer)  # returns padded input_ids and attention_masks.
    label_outputs = batch_padding(labels, tokenizer, pad_token_id=IGNORE_INDEX)
    outputs["labels"] = label_outputs["input_ids"]
    # print(">>>>", torch.Tensor(outputs["input_ids"]).shape, torch.Tensor(outputs["labels"]).shape)
    return {
        "input_ids": torch.Tensor(outputs["input_ids"]).long(),
        "labels": torch.Tensor(outputs["labels"]).long(),
        "attention_mask": torch.Tensor(outputs["attention_mask"]).float()
    }

def test_data_collector(
        batch, tokenizer, 
        src_lang_code="zh", trg_lang_code="en", 
        show_info: bool = False
    ):
    # for raw data lines
    input_ids = []
    if show_info:
        print_once(f">>> batch INFO:")
        print_once(batch)
    
    for item in batch:
        trans_prompt = "Please translate the <src_lan> into <trg_lan>: <src_sent> "
        query =  trans_prompt.replace("<src_lan>", lang_codes.get_lang(src_lang_code)).replace("<trg_lan>",lang_codes.get_lang(trg_lang_code))     
        if lang_codes.get_lang(src_lang_code) == "Chinese":
            item = item.replace(" ", "").strip()
        query = query.replace("<src_sent>", item) + LABEL_MARK
        input_ids.append(
            [tokenizer.bos_token_id] + tokenizer.encode(query, add_special_tokens=False)
        )

    outputs = batch_padding(
        input_ids, tokenizer,max_length=tokenizer.model_max_length-256)  # returns padded input_ids and attention_masks.

    return {
        "input_ids": torch.Tensor(outputs["input_ids"]).long(),
        "attention_mask": torch.Tensor(outputs["attention_mask"]).float()
    }

def gen_rank_pair(df:DataFrame):
    """
    return the ranked data frame by the following dictionary
    cpo_dataset_dict = {
        "prompt": [
            "hello",
            "how are you",
            "What is your name?",
            "What is your name?",
            "Which is the best programming language?",
            "Which is the best programming language?",
            "Which is the best programming language?",
        ],
        "chosen": [
            "hi nice to meet you",
            "I am fine",
            "My name is Mary",
            "My name is Mary",
            "Python",
            "Python",
            "Java",
        ],
        "rejected": [
            "leave me alone",
            "I am not fine",
            "Whats it to you?",
            "I dont have a name",
            "Javascript",
            "C++",
            "C++",
        ],
    }
    """
    preferred_list = []
    lesser_list = []
    prompts_list = []
    src_lang_codes = []
    trg_lang_codes = []

    inputs_list = [i for i in df["input"]]
    values_list = [np.array(eval(i)) for i in df["values"]]
    probs_list = [np.array(eval(i)) for i in df["scores"]]
    sequences_list = [ eval(i) for i in df["sequences"]]
    src_code_list = [i for i in df["src_lang_code"]]
    trg_code_list = [i for i in df["trg_lang_code"]]
    for input, values, sequences, probs, src_lang_code,trg_lang_code in zip(inputs_list, values_list, sequences_list, probs_list, src_code_list, trg_code_list):
        # print(sequences, len(sequences))
        # print(values, len(values))
        assert len(sequences) == len(values), "must be same length"
        for index  in range(len(sequences)):
            for j in range(index, len(sequences)):
                # this pair needs tuning
                if values[index]>values[j] and probs[index]<probs[j]:
                    prompts_list.append(input)
                    src_lang_codes.append(src_lang_code)
                    trg_lang_codes.append(trg_lang_code)
                    preferred_list.append(sequences[index])
                    lesser_list.append(sequences[j])
    out_data = {}
    out_data["prompt"] = prompts_list
    out_data["src_lang_code"] = src_lang_codes
    out_data["trg_lang_code"] = trg_lang_codes
    out_data["chosen"] = preferred_list
    out_data["rejected"] = lesser_list
    out_df = DataFrame(out_data)
    out_df = out_df.drop_duplicates().dropna()
    out_df.reset_index(drop=True, inplace=True) # remove repeat and NaN data 
    return out_df

def build_multilingual_dataloader(
    lang_codes: List[str],
    nas_base_path: str = "",
    batch_size: int = 10,
    num_workers: int = 0,
    distributed: bool = True,
):
    lang_dataloaders = dict()
    for lang in lang_codes:
        fpath = os.path.join(nas_base_path, f"dataset/monolingual/{lang}/merged.txt")
        with open(fpath, "r") as f:
            data = f.readlines()
        if distributed:
            sampler = DistributedSampler(data, shuffle=True)
        else:
            sampler = None
        dataloader = DataLoader(
            data,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=num_workers,
        )
        lang_dataloaders[lang] = dataloader
    return lang_dataloaders
