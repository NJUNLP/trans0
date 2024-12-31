# -*- coding: utf-8 -*-
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from peft import PeftModel
from vllm import LLM, SamplingParams
from configs.prompts import TRANS_PROMPTS, LABEL_MARK, make_mt_instruction
from configs.lang_codes import LangCodes
from utils.common_utils import print_once, set_special_tokens, check_available_memory, get_path
from modules.data import read_rawline_data
from torch.utils.data import DataLoader
from tqdm import tqdm
from modules.data import test_data_collector

import torch.distributed as dist
import torch, time
import gc, os, glob

lang_codes = LangCodes()

def prepare_vllm_inference(args, model_dir=None,override_cache=True, cache_suffix=""):
    """
    build vllm for lora adapted LLM
    """
    target_model_path = get_path(args, args.output_dir) if model_dir is None else model_dir
    print_once(f">>> loading from >>>:{target_model_path}" )
    
    available_gpu = check_available_memory(device_index=0)
    gpu_utilization = min(35.0/available_gpu, 0.9)
    if args.use_lora:
        if override_cache or not os.path.exists(os.path.join(get_path(args, args.cache_dir), "cache_merged_llm"+cache_suffix)):
            base_model = AutoModelForCausalLM.from_pretrained(
                args.llm_path, trust_remote_code=True, device_map="auto")
            peft_model = PeftModel.from_pretrained(base_model, target_model_path)
            merged_model = peft_model.merge_and_unload()
            merged_model.save_pretrained(os.path.join(args.cache_dir, "cache_merged_llm"))
            merged_model.to("cpu")
            del(merged_model)
            gc.collect()
            torch.cuda.empty_cache()
            print_once("release gpu")
        llm = LLM(
            model=os.path.join(get_path(args, args.cache_dir), "cache_merged_llm"), dtype=torch.bfloat16 if args.bf16 else torch.float16,
            tokenizer=target_model_path,  # the finetuned vocabulary
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=gpu_utilization,
        )
    else: # direct loading from the checkpoint
        print("preparing vllm with ", torch.cuda.device_count())
        llm = LLM(
            model = target_model_path, dtype=torch.bfloat16 if args.bf16 else torch.float16,
            tokenizer= target_model_path,  # the finetuned vocabulary
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=gpu_utilization,
        )
        print("done")
    return llm

def vllm_inference_onair(args, override_cache=False):
    """
    :takes inputs from python console and infer on the fly
    """
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
    sampling_params = SamplingParams(n=1, temperature=0., max_tokens=args.max_new_tokens)
    llm = prepare_vllm_inference(args, override_cache=override_cache)
    with torch.no_grad():
        while True:
            print("please input the query:")
            input_l = input()
            tokenizer = llm.get_tokenizer()
            if tokenizer.chat_template is not None:
                input_l = tokenizer.apply_chat_template(
                    make_mt_instruction(input_l), tokenize=False, add_generation_prompt=True)
            generation_out = llm.generate([input_l], sampling_params)
            for item in generation_out:
                for item_out in item.outputs:
                    # l = item_out.text.replace("\n", " ").strip()
                    l = item_out.text
                    if LABEL_MARK in l:
                        mark_index=l.index(LABEL_MARK)
                        print(l.strip()[mark_index:].replace(LABEL_MARK, ""))
                    else:
                        print(">>>> "+l+"\n")


def vllm_inference(args, inputs_list, src_lang_code, trg_lang_code, override_cache=False):
    """
    :param input_lists: inputs are raw lines ["line1", "line2",...]
    """
    trans_prompt = TRANS_PROMPTS[0]
    input_ls = [
        trans_prompt.format(
            src_lan=lang_codes.get_lang(src_lang_code),
            trg_lan=lang_codes.get_lang(trg_lang_code),
            src_sent=l)
        for l in inputs_list
    ]
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"]="spawn"
    sampling_params = SamplingParams(
        n=1, temperature=0, max_tokens=args.max_new_tokens)
    # reload the LLM ckpt (the transformer repo)
    llm = prepare_vllm_inference(
        args, model_dir=None, 
        override_cache=override_cache)
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is not None:
        input_ls = [
            tokenizer.apply_chat_template(
                    make_mt_instruction(l), tokenize=False,
                    add_generation_prompt=True)
            for l in input_ls
        ]
    else:
        input_ls =[
            l+LABEL_MARK for l in input_ls
        ]
    generation_out = llm.generate(
        input_ls, sampling_params=sampling_params)
    return generation_out

def distributed_inference(args, dir, input_lists, src_lang_code, trg_lang_code, override_cache=False):
    """
    :param input_lists: inputs are raw lines ["line1", "line2",...]
    :param dir: the model for validation, default using the model in args.output_dir
    """
    # if args.test_data_path is not None:
    #     cache_path = os.path.join(get_path(args, args.cache_dir), args.test_data_path.split("/")[-1].strip())
    # else:
    #     cache_path = os.path.join(get_path(args, args.cache_dir), args.dev_data_path.split("/")[-1].strip())
    cache_path = os.path.join(get_path(args, args.cache_dir), "cached_inference")
    if override_cache:
        os.system(f"rm -rf %s"%cache_path)
    try:
        os.makedirs(cache_path, exist_ok=True)
    except FileExistsError:
        pass

    target_model_path = get_path(args, args.output_dir) if dir is None else dir
    print_once(f">>> validate trg output_dir >>>:{target_model_path}" )
    # reload the LLM ckpt from the output_dir
    tokenizer = AutoTokenizer.from_pretrained(
        target_model_path,
        model_max_length=args.max_length,
        padding_side = "left",
        truncation_size = "left",
        trust_remote_code=True)
    time.sleep(int(os.environ["ARNOLD_WORKER_NUM"])*10)
    llm = AutoModelForCausalLM.from_pretrained(
        target_model_path, trust_remote_code=True,
        use_cache=True#, device_map=f"cuda:{dist.get_rank()}"
    ).to("cuda")

    # llm.is_parallelizable=True
    # llm.model_parallel=True
    llm, tokenizer = set_special_tokens(llm, tokenizer)  # set special tokens
    llm.eval()  # evaluation mode
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        num_return_sequences=1,
    )  # for huggingface generation

    # specific test dataset by distributed sampler for evaluation
    sampler = torch.utils.data.distributed.DistributedSampler(input_lists, shuffle=False)

    data_loader = DataLoader(
        input_lists, shuffle=False,
        batch_size=args.per_device_eval_batch_size,
        sampler=sampler)
    dist_outs = []
    progress_bar = tqdm(range(len(data_loader)), disable=(dist.get_rank() != 0))
    for _, batch_lines in enumerate(data_loader):
        progress_bar.update(1)
        processed_batch = test_data_collector(
            batch_lines,
            tokenizer=tokenizer,
            src_lang_code=src_lang_code,
            trg_lang_code=trg_lang_code,
        )
        input_ids = processed_batch["input_ids"].to(llm.device)
        with torch.no_grad():
            generation_out = llm.generate(
                input_ids=input_ids,
                attention_mask=processed_batch["attention_mask"].to(llm.device),
                generation_config=generation_config, return_dict_in_generate=True
            )
        output_seq =generation_out.sequences.reshape(
            input_ids.shape[0], generation_config.num_return_sequences, -1)
        input_length = input_ids.shape[1]
        output_seq = output_seq[:, :, input_length:]
        for out_l in output_seq:
            processed_out = tokenizer.batch_decode(out_l, skip_special_tokens=True)[0].replace("\n", " ")
            dist_outs.append(processed_out)

    with open(os.path.join(cache_path, f"rank_{dist.get_rank()}" ), "w") as cache_file:
        print(">>>> cache to rank", dist.get_rank())
        for l in dist_outs:
            cache_file.write(l+ "\n")
    torch.cuda.empty_cache()
    dist.barrier()   # wait for all threads to finish

    merged_results = []
    if dist.get_rank()==0:  # merge by the first thread
        # collect files
        cache_paths = glob.glob(os.path.join(cache_path, "rank_*"))
        sorted_paths = sorted(cache_paths, key=lambda x:int(x.split("rank_")[1]))
        results_for_each_file = []
        for res_path in sorted_paths:
            new_results = read_rawline_data(res_path)
            results_for_each_file.append(new_results)
        while True:
            for sublist in results_for_each_file:
                if sublist:
                    if len(merged_results)>= len(input_lists):
                        break
                    merged_results.append(sublist[0])
                    sublist.pop(0)
                else:
                    break
            if len(merged_results)>=len(input_lists) or all(not sublist for sublist in results_for_each_file):
                break
    return merged_results
