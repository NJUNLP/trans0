# -*- coding: utf-8 -*-
import copy
import gc
import os
import random

import numpy as np
import torch

IGNORE_INDEX=-100

SEP_TOKEN="<sep>"

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

def print_once(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)

def free_gpu():
    gc.collect()
    torch.cuda.empty_cache() # release torch objects

def check_available_memory(device_index:int):
    device = torch.device(f"cuda:{device_index}")
    return (torch.cuda.get_device_properties(device).total_memory -torch.cuda.memory_allocated(device))/1024./1024./1024.

def set_special_tokens(model, tokenizer, show_info=False):
    if tokenizer.pad_token is None and tokenizer.pad_token_id is None:
        print_once(f"WARNING: the pad token of the tokenizer is None")
        # We do not resize the vocab embedding, since it ruins the KL value with the ref_model
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        # tokenizer.pad_token = tokenizer.decode(0)
        print_once(f">>> set pad token to {tokenizer.pad_token}")
        print_once(f">>> set pad token id to {tokenizer.pad_token_id}")

    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.eos_token_id = tokenizer.eos_token_id

    if show_info:
        print_once(tokenizer)

    return model, tokenizer

def compute_loglikelihood(logits, labels):
    """ compute the loglikelihood
    """
    batch_size, seq_length, vocab_size = logits.shape

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)

    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = loss_fct(shift_logits, shift_labels).reshape(batch_size, -1) # [bs * seq_len]
    ignore_mask = labels != IGNORE_INDEX

    avg_loss = loss.sum(dim=-1) / ignore_mask.sum(dim=-1)

    return - avg_loss

def SFTwithKLTrainer(Trainer):
    """ with original alpaca instruct tuning data for KL norm override
        the loss function with additional KL norm by a reference
    """
    def compute_loss(self, model, inputs, return_outputs=False):
        # print_once(f"check inputs: {inputs}")  # debug infos
        model_outputs = model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask']
        )
        logprob = compute_loglikelihood(model_outputs.logits, inputs['labels'])

def aggregate_rejection(seq: str)->str:
    # create a significant rejection
    degradation = ["repeat", "drop","none", "none", "none","none"]
    action = random.choice(degradation)
    out_seq = ""
    if action=="repeat":
        if len(seq)<10:
            out_seq = f"{seq} {seq}"
        else: # repeat the middle part
            start=random.randint(0,len(seq)//2 - 1)
            end=random.randint(start+1, len(seq))
            out_seq = f"{seq[:start]} {seq[start:end]} {seq[start:end]} {seq[end:]}"
    elif action == "drop" and len(seq)>10:
        start = random.randint(0, len(seq)//2 - 1)
        end = random.randint(start+1, len(seq))
        out_seq = seq[:start] + seq[end:]
    else: # nothing changed
        out_seq = copy.deepcopy(seq)
    return out_seq

def get_path(args, path:str)->str:
    if path.startswith("/"):  # absolute path, no need to process
        return path
    else:  # relative path (based by nas_base_path)
        return os.path.join(args.nas_base_path, path)

def get_ranks(number_array):
    """
    return the array of rank
    """
    ascending_index = number_array.argsort()
    rank = np.zeros_like(ascending_index)
    for r in range(len(ascending_index)):
        rank[ascending_index[r]] = r
    return rank

def truncate_encoded(inputs, max_length=500):
    """
    truncate the encoded inputs to max_length:
    inputs: dict of encoded inputs
        keys = ["input_ids","token_type_ids","attention_mask"]
    max_length: 500 for BERT default max_len=512
    """
    trunc_inputs = {"input_ids":inputs["input_ids"][:, :max_length],
                  "token_type_ids":inputs["token_type_ids"][:, :max_length],
                  "attention_mask":inputs["attention_mask"][:, :max_length]
                 }
    return trunc_inputs
