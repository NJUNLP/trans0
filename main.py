# -*- coding: utf-8 -*-
import torch.distributed as dist
import torch
import transformers 
import os, datetime, time, glob, random
import numpy as np
import pandas as pd
import pyarrow as pa
import copy, wandb

from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import free_gpu, print_once, set_special_tokens, get_path
from utils.unit_test import unit_test
from modules.data import (
    get_dataset,
    sft_data_collactor,
    read_json_or_jsonl_data,
    read_parallel_data,
    build_multilingual_dataloader,
)
from datasets import Dataset, load_dataset
from modules.inference import vllm_inference, distributed_inference, vllm_inference_onair
from modules.agent import TransAgent
from configs.configs import DefaultTrainingArguments, peft_config
from configs.prompts import LABEL_MARK, TRANS_PROMPT

from peft import get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP

from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
import comet
"""
sft a huggingface LLM
"""
# os.environ["NCCL_P2P_DISABLE"]="1"  # nccl communicate through shared memory to avoid port fail.

os.environ["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"

def sft_LLM(args, force_lora=False):
    # build dataset
    train_dataset = get_dataset(get_path(args, args.train_data_path), show_info=args.debug_mode)

    # reload LLM, the tokenizer and model used for training
    tokenizer = AutoTokenizer.from_pretrained(
        get_path(args, args.llm_path), 
        model_max_length=args.max_length,  # controls the maximum PE
        padding_side = args.padding_side,
        truncation_size = args.truncation_side,
        trust_remote_code=True
    )
    llm = AutoModelForCausalLM.from_pretrained(get_path(args, args.llm_path), trust_remote_code=True)
    llm, tokenizer = set_special_tokens(llm, tokenizer, show_info=args.debug_mode)
    if args.use_lora or force_lora:  # always lora for initialization
        llm = get_peft_model(llm, peft_config=peft_config)
    llm.config.use_cache= False
    print_once(args)

    llm.is_parallelizable=True
    llm.model_parallel=True

    trainer = Trainer(
        model=llm,
        tokenizer=tokenizer, 
        args=args,
        train_dataset=train_dataset,
        data_collator=lambda x: sft_data_collactor(x, tokenizer, show_info=args.debug_mode)
    )
    train_results = trainer.train(
        resume_from_checkpoint=True if os.path.exists(os.path.join(get_path(args, args.output_dir), "trainer_state.json")) else None
    )
    metrics = train_results.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if dist.get_rank()==0:
        if args.use_lora or force_lora:
            trainer.save_model(output_dir=get_path(args, args.cache_dir))  # cache the lora adaptor for debug
            llm = llm.merge_and_unload()
        llm.save_pretrained(get_path(args, args.output_dir),safe_serialization=True)
        tokenizer.save_pretrained(get_path(args, args.output_dir))
    
    trainer.accelerator.free_memory() # memory leak: release the gpu by accelerator! 
    del llm, tokenizer, train_dataset, train_results, trainer
    free_gpu()
    return

def test(args, use_vllm=False):
    """ fast inference by vllm or multi-thread inference then merge
    :param use_vllm:
    if true, will merge the llm with adaptor in cache_dir for vllm inference (not compatible with 'torchrun' launch)
    
    if false, will distribute the test samples across devices for transformers' inference (launched by torchrun)
    the individual results are cached and merged.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    input_lists = read_json_or_jsonl_data(get_path(args, args.test_data_path))
    # input_lists = input_lists[:9]
    # for l in input_lists:
    #     print(l)
    # print(">>>>file len: ", len(input_lists))
    if use_vllm:
        generation_out = vllm_inference(args, input_lists, src_lang_code="zh", trg_lang_code="en", override_cache=True)
        with open(get_path(args, args.test_data_path)+".out", "w",encoding="utf-8") as out_file:
            for item in generation_out:
                l = item.outputs[0].text.replace("\n", " ").strip()
                if LABEL_MARK in l:
                    mark_index=l.index(LABEL_MARK)
                    out_file.write(l.strip()[mark_index:].replace(LABEL_MARK, "")+"\n")
                else:
                    out_file.write(l.strip()+"\n") 
    else:
        merged_results = distributed_inference(args, input_lists, src_lang_code="zh", trg_lang_code="en", override_cache=True)
        if dist.get_rank()==0:
            with open(get_path(args, args.test_data_path)+".out", "w",encoding="utf-8") as out_file:
                for l in merged_results:
                    if LABEL_MARK in l:
                        mark_index=l.index(LABEL_MARK)
                        out_file.write(l.strip()[mark_index:].replace(LABEL_MARK, "")+"\n")
                    else:
                        out_file.write(l.strip()+"\n") 
    return

def validate(args, dir=None, global_step=None, src_lang_code="zh", trg_lang_code="en"):
    """ 
    validate the parallel data given a csv or parquet from args.dev_data_path
    
    if false, will distribute the test samples across devices for transformers' inference (launched by torchrun)
    the individual results are cached and merged.

    log the validation by global_step when it's not None
    
    """
    assert args.dev_data_path.endswith(".parquet"), "must validate on parquet (parallel data)"
    input_list, reference_list = read_parallel_data(
        data_path=get_path(args, args.dev_data_path), 
        src_lang_code=src_lang_code, trg_lang_code=trg_lang_code)
    merged_results = distributed_inference(args, dir, input_list, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code, override_cache=True)
    if dist.get_rank()==0:  # cache the merged translation to .out file
        processed_out_list = []
        cache_path = os.path.join(get_path(args, args.cache_dir), args.dev_data_path.split("/")[-1].strip())
        with open(os.path.join(cache_path,"merged.out"), "w", encoding="utf-8") as out_file:
            for l in merged_results:
                if LABEL_MARK in l:
                    mark_index=l.index(LABEL_MARK)
                    out_file.write(l.strip()[mark_index:].replace(LABEL_MARK, "")+"\n")
                else:
                    out_file.write(l.strip()+"\n") 
        with open(os.path.join(cache_path,"merged.out"), "r", encoding="utf-8") as out_file:
            for l in out_file:
                processed_out_list.append(l.strip())   
        # evaluate with bleurt
        with torch.no_grad():
            bleurt_scorer = BleurtForSequenceClassification.from_pretrained(get_path(args, args.bleurt_ckpt), device_map="cuda:0", trust_remote_code=True)
            bleurt_tokenizer = BleurtTokenizer.from_pretrained(get_path(args, args.bleurt_ckpt), device_map="cuda:0", trust_remote_code=True)
            bleurt_scorer.eval()
            inputs = bleurt_tokenizer(reference_list, processed_out_list, padding='longest', return_tensors='pt').to("cuda:0")
            res = bleurt_scorer(**inputs).logits.flatten().tolist()
            del bleurt_scorer, bleurt_tokenizer
            free_gpu()

            comet_scorer = comet.load_from_checkpoint(get_path(args, args.comet_ckpt), reload_hparams=True)
            data = []
            for src, mt in zip(input_list, processed_out_list):
                data.append({"src":src, "mt": mt})
            comet_output = comet_scorer.predict(data, batch_size=8, gpus=1)
            bleurt_score = np.array(res).mean()
            comet_score = comet_output.system_score
            print("bleurt=%.4f"%bleurt_score)
            print("comet=%.4f"%comet_score)
            if global_step is not None:
                print(f"bleurt= {format(bleurt_score, '.4f')}, comet= {format(comet_score, '.4f')}")
                wandb.log({
                    "bleurt": bleurt_score, "comet": comet_score,
                    "step": global_step})
            del comet_scorer
    free_gpu()
    return

def self_play(
    args,
    train_round: int,
    trg_lang_codes=[
        "zho_Hans",
        "eng_Latn",
        "deu_Latn",
        "rus_Cyrl",
        "arb_Arab",
        "isl_Latn",
        "kor_Hang",
        "ita_Latn",
    ],
):
    """
    collect the preference data via self-play on specific lang_pair, data is cached as csv
    the default trg_lang is english
    src_lang_code is a list of lang_codes 
    """
    node_rank = dist.get_rank()
    def get_dataloader_for_round():
        lang_idx = (train_round + node_rank) % len(trg_lang_codes)
        lang = trg_lang_codes[lang_idx]
        dataloader = multilingual_dataloader[lang]
        sampler = dataloader.sampler
        if sampler is not None:
            sampler.set_epoch(train_round)
        return dataloader, lang
    random.seed(int(time.time())+node_rank)
    lang_dataloader, src_lang_code = get_dataloader_for_round()
    lang_code = random.choice([l for l in trg_lang_codes if l != src_lang_code])
    for batch in lang_dataloader:
        src_list = batch
        break

    # initialize the translation agent for self-play data collection
    agent = TransAgent(args)  # initiate a MC agent with auto mapping(distributed) to generate data. # requires the training data path 
    # agent.distributed_valued_by_mcts(src_list, src_lang_code=src_lang_code, trg_lang_code="en")
    agent_mct_df = []
    for line in src_list:
        mc_tree = agent.MCTS(src_sent=line.strip(), src_lang_code=src_lang_code, trg_lang_code=lang_code)
        agent_mct_df.append(agent.yield_tree2rank(mc_tree))
        # agent.valued_by_BLEUrt(src_list, trg_list, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code)  # for tuning
    local_df = pd.concat(agent_mct_df, ignore_index=True)
    save_path = os.path.join(
        agent.cache_dir,
        f"{src_lang_code}-{lang_code}.{dist.get_rank()}.self_play_{str(train_round)}.csv",
    )
    local_df.to_csv(save_path, index=False)

    dist.barrier()
    if dist.get_rank()==0:
        collected_df = []
        df_path = glob.glob(os.path.join(agent.cache_dir, f"*-*.*.self_play_{str(train_round)}.csv"))
        for file in df_path:
            distributed_df = pd.read_csv(file)
            for i in range(len(distributed_df)):
                translate_prompt = random.choice(TRANS_PROMPT)
                in_line = distributed_df.at[i, 'prompt']
                src_code = distributed_df.at[i, 'src_lang_code']
                trg_code = distributed_df.at[i, 'trg_lang_code']
                distributed_df.at[i, "prompt"] = (
                    translate_prompt.replace(
                        "<src_lan>", agent.supported_langs.get_lang(src_code)
                    )
                    .replace("<trg_lan>", agent.supported_langs.get_lang(trg_code))
                    .replace("<src_sent>", in_line)
                )
            collected_df.append(distributed_df)
        merged = pd.concat(collected_df, ignore_index=True)
        merged = merged.drop_duplicates().dropna()
        merged.reset_index(drop=True, inplace=True)
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d")
        merge_fpath = os.path.join(
            agent.cache_dir, f"self_play_{str(train_round)}.{time_stamp}.csv"
        )
        merged.to_csv(merge_fpath, index=False)
    del agent, src_list, local_df
    for item in agent_mct_df:
        del item
    free_gpu()
    return

def RL_update(args, train_round:int):
    agent = TransAgent(args, train=train_round)  # initiate a MC agent for update # requires the training data path 
    cached_files = glob.glob(os.path.join(agent.cache_dir, f"self_play_{str(train_round)}*.csv"))
    cached_SP_dir = sorted(cached_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
    merged_df = pd.read_csv(cached_SP_dir)
    tuning_dataset = Dataset(pa.Table.from_pandas(merged_df))
    print("loading RL finetune data.")
    start=time.time()
    agent.update_policy(tuning_dataset) 
    end = time.time()

    del agent, tuning_dataset, merged_df
    free_gpu()
    print(">> lapse >>:", end-start)
    return


if __name__=="__main__":
    random.seed(int(time.time()))
    parser = transformers.HfArgumentParser(DefaultTrainingArguments)  # subclass of ArgumentParser
    parser.add_argument(
        "-m", "--mode", type=str, default='SFT',
        choices=['SFT', 'RL', "test", "valid", "air", "simulate"],
        help="SFT (imitation learning with KL div) or RL"
    )
    parser.add_argument("--src_code", type=str, default="zh", help="indicate src language type for validation")
    parser.add_argument("--trg_code", type=str, default="en", help="indicate trg language type for validation")
    args = parser.parse_args()  # inject add_argument parts

    os.environ["HF_HOME"] = os.path.join(args.nas_base_path, "cache")
    os.environ["HF_DATASETS_CACHE"]=os.path.join(args.nas_base_path, "cache")
    os.environ["NCCL_DEBUG"]="INFO"
    if args.mode=="SFT":
        # load_dataset(args.flores_script,"all", trust_remote_code=True)
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        sft_LLM(args)
    elif args.mode== "test":
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        test(args)
    elif args.mode== "valid":
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
        src_lan_code = args.src_code
        trg_lan_code = args.trg_code
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        validate(args, src_lang_code=src_lan_code, trg_lang_code=trg_lan_code)
    elif args.mode=="air":
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        vllm_inference_onair(args, override_cache=True)
    elif args.mode== "RL":
        # load_dataset(args.flores_script,"all", trust_remote_code=True)
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        sft_LLM(args, force_lora=True)
        wandb.finish()
        dist.barrier()
        if dist.get_rank()==0:
            wandb.init()
            wandb.define_metric("bleurt", step_metric="step")
            wandb.define_metric("comet", step_metric="step")
        validate(
            args, global_step=0,
            src_lang_code="eng_Latn", trg_lang_code="arb_Arab"
        )
        trg_lang_codes = ["zho_Hans", "eng_Latn", "isl_Latn", "arb_Arab"]
        global multilingual_dataloader
        multilingual_dataloader = build_multilingual_dataloader(
            trg_lang_codes, args.nas_base_path, batch_size=10
        )
        for train_round in range(200):
            self_play(args, train_round, trg_lang_codes=trg_lang_codes)
            dist.barrier()
            RL_update(args, train_round)
            dist.barrier()
            validate(
                args, dir=os.path.join(get_path(args,args.output_dir), "_RL"), 
                global_step=train_round+1,
                src_lang_code="eng_Latn", trg_lang_code="arb_Arab"
            )

    elif args.mode== "simulate":
        args = args = parser.parse_args_into_dataclasses()[0]
        unit_test(args)
    else:
        print(">>> undefined mode, exit")
        
        
