# -*- coding: utf-8 -*-
import torch.distributed as dist
import torch
import transformers
import os, datetime, time, glob, random
import numpy as np
import pandas as pd
import pyarrow as pa
import wandb

from vllm import SamplingParams
from transformers import Trainer, AutoTokenizer, AutoModelForCausalLM
from utils.common_utils import free_gpu, print_once, set_special_tokens, get_path
from utils.infer_utils import process_flores_test, process_mix_flores_test, extract_test
from utils.unit_test import unit_test
from modules.data import (
    get_dataset,
    sft_data_collactor,
    read_rawline_data,
    read_parallel_data,
    build_multilingual_dataloader,
)
from datasets import Dataset
from modules.inference import (
    lang_codes,
    vllm_inference,
    distributed_inference,
    vllm_inference_onair,
    prepare_vllm_inference,
)
from modules.agent import TransAgent
from modules.metrics import BleurtScorer, CometScorer
from configs.configs import DefaultTrainingArguments, peft_config
from configs.prompts import LABEL_MARK, TRANS_PROMPTS, make_mt_instruction

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

def test(args, src_lang_code, trg_lang_code, use_vllm=False):
    """ fast inference by vllm or multi-thread inference then merge
    :param use_vllm:
    if true, will merge the llm with adaptor in cache_dir for vllm inference (not compatible with 'torchrun' launch)

    if false, will distribute the test samples across devices for transformers' inference (launched by torchrun)
    the individual results are cached and merged.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    input_lists = read_rawline_data(get_path(args, args.test_data_path))
    # input_lists = input_lists[:9]
    # for l in input_lists:
    #     print(l)
    # print(">>>>file len: ", len(input_lists))
    output_file_name = args.test_data_path.split("/")[-1].split(".")[0]

    if use_vllm:
        generation_out = vllm_inference(args, input_lists, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code, override_cache=True)
        with open(output_file_name +".out", "w",encoding="utf-8") as out_file:
            for item in generation_out:
                l = item.outputs[0].text.replace("\n", " ").strip()
                if LABEL_MARK in l:
                    mark_index=l.index(LABEL_MARK)
                    out_file.write(l.strip()[mark_index:].replace(LABEL_MARK, "")+"\n")
                else:
                    out_file.write(l.strip()+"\n")
    else:
        merged_results = distributed_inference(args, input_lists, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code, override_cache=True)
        if dist.get_rank()==0:
            with open(output_file_name + ".out", "w",encoding="utf-8") as out_file:
                for l in merged_results:
                    if LABEL_MARK in l:
                        mark_index=l.index(LABEL_MARK)
                        out_file.write(l.strip()[mark_index:].replace(LABEL_MARK, "")+"\n")
                    else:
                        out_file.write(l.strip()+"\n")
    return

def validate_pair(args,  flores_script, src_lang_code, trg_lang_code, model_dir=None, global_step=None):
    """
    validate the parallel from flores scripts.
    extract and cache parallel valid set for validation.

    log the validation by global_step when it's not None
    :param model_dir: the ckpt to validate
    """
    # generate the inference data by the flores.py script
    valid_file_name = f"flores_test_{src_lang_code}-{trg_lang_code}.parquet"
    valid_data_dir = os.path.join(get_path(args, args.cache_dir), valid_file_name)
    if not os.path.exists(valid_data_dir):
        process_flores_test(flores_script, src_lang_code, trg_lang_code, valid_data_dir)
    print_once(f">>>> valid {valid_data_dir}...")
    input_list, reference_list = read_parallel_data(
        data_path=valid_data_dir,
        src_lang_code=src_lang_code, trg_lang_code=trg_lang_code)
    merged_results = distributed_inference(args, model_dir, input_list, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code, override_cache=True)
    if dist.get_rank()==0:  # cache the merged translation to .out file
        processed_out_list = []
        cache_path = os.path.join(get_path(args, args.cache_dir), "cached_inference")
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
            bleurt_scorer=BleurtScorer(ckpt_path=get_path(args, args.bleurt_ckpt))
            bleurt_score = bleurt_scorer.score(reference_list, processed_out_list)
            del bleurt_scorer
            free_gpu()

            comet_scorer = CometScorer(ckpt_path=get_path(args, args.comet_ckpt))
            comet_score = comet_scorer.score(input_list, processed_out_list)
            del comet_scorer
            free_gpu()

            print("bleurt=%.4f"%bleurt_score)
            print("comet=%.4f"%comet_score)
            if global_step is not None:
                print(f"bleurt= {format(bleurt_score, '.4f')}, comet= {format(comet_score, '.4f')}")
                wandb.log({
                    "bleurt": bleurt_score, "comet": comet_score,
                    "step": global_step})

    free_gpu()
    return

def validate(args, valid_type:str="en2x",dev_data_path=None,model_dir=None, global_step=None):
    """
    validate by dev_data_path file, log the validation by global_step when it's not None
    :param valid_type: the type of the validation, ["en-x", "x-x", "x-en", "all"]
    :param dev_data_path: a parallel data file. If None, will extract flores.py for multi-lingual parallel test
    :param valid_type: the type of the validation, ["input_lang_code", "input", "output_lang_code", "output"]
    :param model_dir: the model to validate, if None, validate the model in args.output_dir
    """
    if not os.path.exists(os.path.join(get_path(args, args.cache_dir))):
        os.makedirs(os.path.join(get_path(args, args.cache_dir)))
    # collect and cache the mixed test-data from flores.py
    valid_file_dir = process_mix_flores_test(
        args.flores_script, args.self_play_languages,
        output_dir= os.path.join(get_path(args, args.cache_dir), "mix_test.parquet")
    )
    print_once(f">>>> valid {valid_file_dir}...")
    extracted_df = extract_test(valid_file_dir, valid_type=valid_type)
    if "ALMA" in args.output_dir:
        trans_prompt = TRANS_PROMPTS[1]
    elif "Tower" in args.output_dir:
        trans_prompt = TRANS_PROMPTS[2]
    else:
        trans_prompt = TRANS_PROMPTS[0]
    raw_inputs = [item["input"] for item in extracted_df]
    input_lists = [
        trans_prompt.format(
            src_lan=lang_codes.get_lang(item["input_lang_code"]),
            trg_lan=lang_codes.get_lang(item["output_lang_code"]),
            src_sent=item["input"]
        )
        for item in extracted_df
    ]
    target_lists = [item["output"] for item in extracted_df]  # cached for metric evaluation

    # prepare VLLM inference
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
    sampling_params = SamplingParams(
        n=1, temperature=0.,
        max_tokens=args.max_new_tokens)
    llm = prepare_vllm_inference(
        args, model_dir=model_dir,
        override_cache=True, cache_suffix=valid_type
    )
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is not None:
        input_lists = [
            tokenizer.apply_chat_template(
                make_mt_instruction(input_l), tokenize=False, 
                add_generation_prompt=True
            ) for input_l in input_lists
        ]
    elif "ALMA" not in args.output_dir:
        input_lists = [
            input_l+LABEL_MARK for input_l in input_lists
        ]
    generation_out = llm.generate(
        input_lists, sampling_params=sampling_params)
    cached_out_lists = [] # cached for metric evaluation
    cached_out_path = os.path.join(get_path(args, args.cache_dir), f"{valid_type}.out")
    with open(cached_out_path, "w", encoding="utf-8") as out_file:
        for item in generation_out:
            for item_out in item.outputs:
                l = item_out.text
                if "</LABEL>" in l:
                    l = l.replace("</LABEL>", "")
                if LABEL_MARK in l:
                    mark_index=l.index(LABEL_MARK)
                    out_file.write(l.strip()[mark_index:].replace(LABEL_MARK, "")+"\n")
                    cached_out_lists.append(l.strip()[mark_index:].replace(LABEL_MARK, ""))
                else:
                    out_file.write(l.strip()+"\n")
                    cached_out_lists.append(l.strip())
    print("finished")
    print(">> test snipet>>", input_lists[0], cached_out_lists[0])
    del llm, sampling_params
    dist.destroy_process_group()
    free_gpu()
    # load the scores
    bleurt_scorer = BleurtScorer(ckpt_path=get_path(args, args.bleurt_ckpt))
    bleurt_score = bleurt_scorer.score(target_lists, cached_out_lists)
    del bleurt_scorer
    free_gpu()

    comet_scorer = CometScorer(ckpt_path=get_path(args, args.comet_ckpt))
    comet_score = comet_scorer.score(raw_inputs, cached_out_lists)
    del comet_scorer
    free_gpu()
    print(valid_type," finished")
    print("bleurt=%.4f"%bleurt_score)
    print("comet=%.4f"%comet_score)

    return

def self_play(
        args, train_round: int,
        trg_lang_codes=[
            "deu_Latn","por_Latn","fra_Latn","ita_Latn",
            "eng_Latn","hin_Deva","spa_Latn","vie_Latn",
            "zho_Hans","rus_Cyrl","ukr_Cyrl", "kor_Hang",
            "arb_Arab","heb_Hebr",
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
                translate_prompt = random.choice(TRANS_PROMPTS[:-1])  # tuning always use the standard mt prompt
                in_line = distributed_df.at[i, 'prompt']
                src_code = distributed_df.at[i, 'src_lang_code']
                trg_code = distributed_df.at[i, 'trg_lang_code']
                distributed_df.at[i, "prompt"] = (
                    translate_prompt.format(
                        src_lan = agent.supported_langs.get_lang(src_code),
                        trg_lan = agent.supported_langs.get_lang(trg_code),
                        src_sent = in_line
                    )
                )
            collected_df.append(distributed_df)
        merged = pd.concat(collected_df, ignore_index=True)
        merged.fillna("", inplace=True) # re-fill the empty line
        # merged.reset_index(drop=True, inplace=True)
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
    merged_df.fillna("", inplace=True)
    tuning_dataset = Dataset(pa.Table.from_pandas(merged_df))
    print("loading RL finetune data.")
    start=time.time()
    rl_loss = agent.update_policy(tuning_dataset)
    end = time.time()
    if dist.get_rank()==0:
        wandb.log({"loss":rl_loss,"step": train_round+1})

    del agent, tuning_dataset, merged_df
    free_gpu()
    print(">> lapse >>:", end-start)
    return


if __name__=="__main__":
    random.seed(int(time.time()))
    parser = transformers.HfArgumentParser(DefaultTrainingArguments)  # subclass of ArgumentParser
    parser.add_argument(
        "-m", "--mode", type=str, default='SFT',
        choices=['SFT', 'RL', "test", "valid", "air", "simulate", "debug_RL"],
        help="SFT (imitation learning with KL div) or RL"
    )
    parser.add_argument("--src_code", type=str, default="all", help="indicate src language type for validation")
    parser.add_argument("--trg_code", type=str, default="all", help="indicate trg language type for validation")
    args = parser.parse_args()  # inject add_argument parts

    os.environ["HF_HOME"] = os.path.join(args.nas_base_path, "cache")
    os.environ["HF_DATASETS_CACHE"]=os.path.join(args.nas_base_path, "cache")
    os.environ["NCCL_DEBUG"]="INFO"
    if args.mode=="SFT":
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        sft_LLM(args)
    elif args.mode== "test":
        src_lan_code = args.src_code
        trg_lan_code = args.trg_code
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        test(args, src_lan_code, trg_lan_code, use_vllm=True)
    elif args.mode== "valid":
        # dist.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        validate(
            args, valid_type="x2en",
        )
    elif args.mode=="air":
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        vllm_inference_onair(args, override_cache=True)
    elif args.mode== "RL":
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
        src_lan_code = args.src_code
        trg_lan_code = args.trg_code
        args = parser.parse_args_into_dataclasses()[0]  # initialize default huggingface parameters
        sft_LLM(args, force_lora=True)
        wandb.finish()
        dist.barrier()
        if dist.get_rank()==0:
            wandb.init()
            wandb.define_metric("loss", step_metric="step")
            wandb.define_metric("bleurt", step_metric="step")
            wandb.define_metric("comet", step_metric="step")
        validate_pair(
            args,flores_script=args.flores_script,
            src_lang_code=src_lan_code, trg_lang_code=trg_lan_code,
            global_step=0
        )
        global multilingual_dataloader
        multilingual_dataloader = build_multilingual_dataloader(
            args.self_play_languages, args.nas_base_path, batch_size=10
        )
        for train_round in range(200):
            self_play(args, train_round, trg_lang_codes=args.self_play_languages)
            dist.barrier()
            RL_update(args, train_round)
            dist.barrier()
            validate_pair(
                args, flores_script=args.flores_script,
                src_lang_code=src_lan_code, trg_lang_code=trg_lan_code,
                model_dir=os.path.join(get_path(args,args.output_dir), "_RL"),
                global_step=train_round+1,
            )
    elif args.mode=="debug_RL":
        dist.init_process_group(backend="nccl", timeout=datetime.timedelta(days=1))
        src_lan_code = args.src_code
        trg_lan_code = args.trg_code
        args = args = parser.parse_args_into_dataclasses()[0]
        dist.barrier()
        if dist.get_rank()==0:
            wandb.init()
            wandb.define_metric("loss", step_metric="step")
            wandb.define_metric("bleurt", step_metric="step")
            wandb.define_metric("comet", step_metric="step")
        for train_round in range(40):
            RL_update(args, train_round)
            dist.barrier()
            validate_pair(
                args, flores_script=args.flores_script,
                src_lang_code=src_lan_code, trg_lang_code=trg_lan_code,
                model_dir=os.path.join(get_path(args,args.output_dir), "_RL"),
                global_step=train_round+1,
            )

    elif args.mode== "simulate":
        args = args = parser.parse_args_into_dataclasses()[0]
        unit_test(args)
    else:
        print(">>> undefined mode, exit")
