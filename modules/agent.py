# -*- coding: utf-8 -*-
import glob
import os
import random
import sys
from typing import List, OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from datasets import Dataset
from lingua import LanguageDetectorBuilder
from pandas import DataFrame
from peft import get_peft_model
from sacrebleu.metrics import BLEU, CHRF
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import DPOConfig, DPOTrainer

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.configs import sp_peft_config
from configs.lang_codes import LangCodes
from configs.prompts import LABEL_MARK, TRANS_CONTEXT_PROMPT, TRANS_PROMPTS, make_mt_instruction
from modules.data import gen_rank_pair
from modules.metrics import BleurtScorer, CometScorer
from modules.NaryTree import *
from utils.common_utils import (
    aggregate_rejection,
    free_gpu,
    get_path,
    print_once,
    set_special_tokens,
)

class TransAgent:
    # wrap the methods and functions for trans0.
    def __init__(self, args, train=None, override_cache=False, metric_type="bleurt"):
        # initiate agent by SFTed LLM for translation.
        self.args = args  # reserve the parameters
        self.self_play_lang_codes = args.self_play_languages
        self.sample_size = args.mcts_sample_size
        self.language_detector = LanguageDetectorBuilder.from_all_languages().build()
        # agent's cache
        self.cache_dir = os.path.join(get_path(args, args.cache_dir), args.output_dir.split("/")[-1], "trans0_agent")
        self.train_count = train
        if train is None:  # override the cache
            if os.path.exists(self.cache_dir) and override_cache:
                os.system(f"rm -rf {self.cache_dir}")
            os.makedirs(self.cache_dir, exist_ok=True)
        else:
            assert os.path.exists(self.cache_dir), f">>>cache required {self.cache_dir} lost!!"
        self.agent_out_dir = os.path.join(get_path(args, args.output_dir),"_RL") # RL tuning savings
        if os.path.exists(os.path.join(self.agent_out_dir,"trainer_state.json")):
            ckpt_path = self.agent_out_dir
        else:
            ckpt_path = get_path(args, args.output_dir) # no exsiting RL tuning
        print_once(f">>>loading the agent from {ckpt_path}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path,
            model_max_length=args.max_length,  # controls the maximum PE
            padding_side = args.padding_side,
            truncation_size = args.truncation_side,
            trust_remote_code=True
        )
        self.base=None  # base is the ref model in DPO training and BT evaluation in MCTS simulation.
        if args.use_lora:
            if train is not None:  # training mode
                llm_base = AutoModelForCausalLM.from_pretrained(
                    ckpt_path, trust_remote_code=True #,device_map=f"cuda:{dist.get_rank()}"
                    ).to("cuda")
                # print(f">> base size: {ckpt_path}>>: ", llm_base.num_parameters()) #
                self.model = get_peft_model(
                    llm_base, peft_config=sp_peft_config)
                # print(f">> reloaded: {ckpt_path}>>: ", self.model.num_parameters()) #
                self.model.print_trainable_parameters()  # the base model is the SFTe-initiated model
                self.base = AutoModelForCausalLM.from_pretrained(
                    get_path(args, args.output_dir), load_in_8bit=False, trust_remote_code=True,
                    torch_dtype=torch.bfloat16)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path, trust_remote_code=True #,device_map=f"cuda:{dist.get_rank()}"
                    ).to("cuda")
                self.base = self.model
        else:
            if train is not None:
                self.model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path, load_in_8bit=False, trust_remote_code=True,
                    torch_dtype=torch.bfloat16)
                self.model.config.use_cache = False
                self.base = AutoModelForCausalLM.from_pretrained(
                    get_path(args, args.output_dir), load_in_8bit=False, trust_remote_code=True,
                    torch_dtype=torch.bfloat16)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path, trust_remote_code=True #, device_map=f"cuda:{dist.get_rank()}"
                    ).to("cuda")
                self.base=self.model
                # self.base = AutoModelForCausalLM.from_pretrained(
                #     args.output_dir, trust_remote_code=True, device_map=f"cuda:{dist.get_rank()}")

        self.model, self.tokenizer = set_special_tokens(self.model, self.tokenizer)
        self.model.is_parallelizable=True
        self.model.model_parallel=True

        self.metric_type = metric_type
        if self.metric_type=="chrf":
            self.scorer = CHRF()
        elif self.metric_type=="bleu":
            self.scorer = BLEU()
        elif self.metric_type=="bleurt":
            self.scorer = BleurtScorer(get_path(args,args.bleurt_ckpt), batch_size=25)
        elif self.metric_type=="comet":
            self.scorer = CometScorer(get_path(args, args.comet_ckpt), batch_size=25)
            self.scorer.load_cuda(self.model.device)
        else:
            raise ValueError("Invalid metric_type. Please choose from 'chrf', 'bleu', or 'bleurt'.")


        self.generate_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.1, do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
        )  # for generation
        self.sample_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.5, do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=self.sample_size,
        )  # for sampling
        total_device_count = dist.get_world_size()
        self.pref_train_config = DPOConfig(
            logging_steps=args.logging_steps,
            max_length=args.max_length,
            max_prompt_length=1024,
            output_dir=self.agent_out_dir,
            learning_rate=args.rl_learning_rate,
            lr_scheduler_type=args.rl_lr_scheduler_type,
            loss_type=args.rl_loss_type, #cpo_alpha=.0,
            per_device_train_batch_size=1,
            deepspeed=args.deepspeed,
            gradient_accumulation_steps=int(args.rl_batch_size//total_device_count),
            resume_from_checkpoint=True,
            save_strategy="no",
            remove_unused_columns=args.remove_unused_columns,
            bf16=args.bf16,
            tf32=args.tf32
        )
        # default translate prompt for MCTS inference and RL update
        if "alma" in args.output_dir.lower():
            self.default_prompt  = TRANS_PROMPTS[1]
        elif "tower" in args.output_dir.lower():
            self.default_prompt  = TRANS_PROMPTS[2]
        else:
            self.default_prompt  = TRANS_PROMPTS[0]

        self.supported_langs = LangCodes()

    def detect_lang(self, text:str)->str:
        """
        detect the language of the input text
        """
        detect_code = self.language_detector.detect_language_of(text)
        if detect_code is None:
            detect_language = "Null_language"
            print(">>>> item language not known:", text)
        else:
            detect_language = self.supported_langs.get_lang(detect_code.iso_code_639_1.name.lower())
        return detect_language

    def score(self, references:List[str], candidates:List[str])->List:
        """
        wraping the score interface to evaluate reconstructions return a list of scores
        """
        if self.metric_type in ["bleurt","comet"]:
            with torch.no_grad():
                # one-on-one bleurt with a list of score output
                res1 = self.scorer.score(references, candidates, keepdims=True)
                res2 = self.scorer.score(candidates, references, keepdims=True)
                res=((res1+res2)/2).tolist()
                return res
        elif self.metric_type in ["chrf", "bleu"]:
            scores = []
            for ref,cand in zip(references, candidates):
                score1=self.scorer.sentence_score(hypothesis=cand, references=[ref]).score
                score2=self.scorer.sentence_score(hypothesis=ref, references=[cand]).score
                scores.append((score2+score1)/2)
            return scores

    # def valued_by_BLEUrt(self,
    #         inputs_list, targets_list,
    #         src_lang_code:str, trg_lang_code:str):
    #     """
    #     used for RL hyperparameter finetuning.
    #     """
    #     llm = self.model
    #     llm.eval()
    #     if dist.get_rank()==0:
    #         with torch.no_grad():
    #             mc_results = []
    #             for src_line, trg_line in zip(inputs_list, targets_list):
    #                 explored_trgs, scores = self.step_explore(
    #                     llm, src_line, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code, sample_mode=False
    #                 )
    #                 # calculate the reconstructed trgs with reference trgs for bleurt
    #                 rewards = self.score(references=[trg_line]*len(explored_trgs), candidates=explored_trgs)
    #                 collected = {"input": src_line, "src_lang_code": src_lang_code, "trg_lang_code": trg_lang_code,
    #                     "sequences":explored_trgs, "scores":scores, "values":rewards}
    #                 print(collected)
    #                 mc_results.append(collected)
    #             free_gpu()
    #             # yield to the distributed cache
    #             MC_df = pd.DataFrame(mc_results) # data frame for RL tuning
    #             MC_df.to_csv(
    #                 os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+f".{dist.get_rank()}"),
    #                 index=False
    #             )
    #             collect_df = pd.read_csv(os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+f".{dist.get_rank()}"))
    #             dict4CPO = gen_rank_pair(collect_df)
    #             for i in range(len(dict4CPO)):  # update the prompts
    #                 translate_prompt = random.choice(TRANS_PROMPT)
    #                 in_line = dict4CPO.at[i, 'prompt']
    #                 src_code = dict4CPO.at[i, 'src_lang_code']
    #                 trg_code = dict4CPO.at[i, 'trg_lang_code']
    #                 dict4CPO.at[i, 'prompt'] = translate_prompt.replace("<src_lan>", self.supported_langs.get_lang(src_code)).replace("<trg_lan>",self.supported_langs.get_lang(trg_code)).replace("<src_sent>", in_line)
    #             dict4CPO.to_csv(
    #                 os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv"),
    #                 index=False
    #             )
    #     dist.barrier()
    #     return os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv")

    # def value_by_MCTS(self,src_sent:str,
    #         src_lang_code:str, trg_lang_code:str,
    #         max_simulation_depth:int):
    #     """
    #     a dev used for RL tuning. more max_simnulation depth for larger ranking variance.
    #     """
    #     llm = self.model
    #     llm.eval()
    #     assert self.supported_langs.check_support(src_lang_code), "source must be supported languages"
    #     assert self.supported_langs.check_support(trg_lang_code), "target must be supported languages"
    #     # the exploration includes the sampled inference by origin prompts and contexted prompts
    #     with torch.no_grad():
    #         explored_trgs, scores = self.step_explore(
    #             llm, src_sent, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code,
    #             sample_mode=False
    #         )
    #         # add the explored trgs to tree and back-translation for semantic rewards
    #         recon_srcs = []
    #         for t_line in explored_trgs:  # evaluate by base model reconstruction
    #             recon_srcs.extend(self.step_explore(
    #                 self.base, t_line, src_lang_code=trg_lang_code, trg_lang_code=src_lang_code,
    #                 sample_mode=False)[0])
    #         rewards_flat = self.score(references=[src_sent]*len(recon_srcs), candidates=recon_srcs)

    #         recon_src_values = self.simulate(  # simulate with base model
    #             self.base, input_list=recon_srcs,
    #             src_lang_code=src_lang_code, trg_lang_code=trg_lang_code,
    #             max_simulation_depth=max_simulation_depth
    #         )
    #         rewards_flat = [(a+b*max_simulation_depth)/(max_simulation_depth+1) for (a,b) in zip(rewards_flat, recon_src_values)]
    #         rewards = np.array(rewards_flat).reshape(-1, self.sample_size).mean(axis=-1).tolist()  # rewards on translation

    #     return {"sequences":explored_trgs,
    #             "scores":scores,
    #             "values":rewards}

    def MCTS(
            self, src_sent:str,
            src_lang_code:str, trg_lang_code:str,show_info=True,
            MC_count:int=20, max_simulation_depth=2,
        )-> NaryTree:
        """
        choose and expand the tree for MC_count rounds, each round will expand a single node.

        collect the data in src_sent language above certain reconstruction threshold.
        each tree node is specified by the corresponding language code.
        :param src_sent: a single sentence
        :param device: a single device vllm object for fast exploration
        :return: a MC tree rooted by the src_sent.
        """
        llm = self.model
        llm.eval()
        assert self.supported_langs.check_support(src_lang_code), "source must be supported languages"
        assert self.supported_langs.check_support(trg_lang_code), "target must be supported languages"
        # the exploration includes the sampled inference by origin prompts and contexted prompts
        with torch.no_grad():
            # fast initiation:
            mc_tree = NaryTree(state={"data":src_sent.strip(), "lang_code":src_lang_code, "recon": None})
            root_lang_code = mc_tree.root.state["lang_code"]
            explored_trgs, scores = self.step_explore(
                llm, mc_tree.root.state["data"], src_lang_code=src_lang_code, trg_lang_code=trg_lang_code,
                sample_mode=True
            )
            # print(">>> explored trgs:",explored_trgs)
            for t_line in explored_trgs: # a dummy simulation for fast initiation.
                recon_srcs = self.step_explore(
                        self.base, t_line, src_lang_code=trg_lang_code, trg_lang_code=root_lang_code,
                        sample_mode=False
                    )[0]
                rewards = self.score(references=[src_sent]*len(recon_srcs), candidates=recon_srcs)
                best_index = np.array(rewards).argmax(axis=-1)
                child = mc_tree.root.add_child(
                    state={"data":t_line, "lang_code":trg_lang_code, "recon": recon_srcs[best_index]}
                )
                mc_tree.backpropagate(child, value=np.array(rewards).mean()) # updates the child upto the root

            # mc tree search
            best_node_previous = mc_tree.root  # top best nodes for context.
            for count in range(MC_count):
                current_node = mc_tree.select()  # select a leaf for expansion.
                if show_info:
                    print(f">>>> {count} node:", current_node.state["data"])
                if best_node_previous == current_node or best_node_previous.state["recon"]==None:
                    # mutation by reconstruction
                    # (explored by given back-translated paraphrase, or whitespace paraphrase)
                    explored_trgs, _ = self.step_explore(
                        llm, current_node.state["recon"],
                        src_lang_code=root_lang_code, trg_lang_code=trg_lang_code,
                        trans_context=None, sample_mode=True
                    )
                else:
                    # merging nodes
                    best_history=[
                        {best_node_previous.state["lang_code"]: best_node_previous.state["data"],
                        root_lang_code: best_node_previous.state["recon"]},
                        {current_node.state["lang_code"]: current_node.state["data"],
                        root_lang_code: current_node.state["recon"]}
                    ]
                    # print(f">>>> history: {best_history}") # expand current node with new child
                    explored_trgs, _ = self.step_explore(
                        llm, src_sent,  #current_node.state["recon"]
                        src_lang_code=root_lang_code, trg_lang_code=trg_lang_code,
                        trans_context=best_history, sample_mode=False
                    )
                recon_srcs, _= self.step_explore(
                    self.base, explored_trgs[0],
                    src_lang_code=trg_lang_code, trg_lang_code=root_lang_code,
                    sample_mode=False
                )
                # simulated_value = self.score(references=[src_sent]*len(recon_srcs), candidates=recon_srcs)[0]
                best_recon, simulated_value = self.simulate(  # simulation via multiple reconstruction circuits.
                    self.base, trg2value=explored_trgs[0], origin_src=src_sent,
                    src_lang_code=root_lang_code, trg_lang_code=trg_lang_code,
                    max_simulation_depth=max_simulation_depth
                )
                new_node = mc_tree.add_child(
                    parent = current_node,
                    child_data={"data":explored_trgs[0], "lang_code": trg_lang_code, "recon":best_recon}
                )
                # print(">>>> new node", new_node.state["recon"], new_node.state["data"])
                # print(">>>> value", simulated_value)
                mc_tree.backpropagate(new_node, simulated_value)
                # update the node record with best utility
                best_node_previous = mc_tree.get_best(mc_tree.root)
        return mc_tree

    def yield_tree2rank(self, mc_tree:NaryTree, threshold=0.0, value_type="utility", sort_type="select")->DataFrame:
        """
        yield the tree results to a ranking dataframe for training
        a MCT in layerswise traversal example (ancesters are ahead of descendants):
        [('落霞与孤鹜齐飞，秋水共长天一色', 0.529447915361208),   # the root
        ('The setting sun and the lone wild goose fly together, the autumn waters blend with the sky in one color.', 0.5596388263834847),
        ('The setting sun and the lone crane fly together, the autumn water is the same color as the sky.', 0.5597622022032738),
        ('The setting sun and the lone wild goose fly together, the autumn water and the sky are of the same color.', 0.5720633864402771)]

        :param value_type: rank by value: ["utility", "value", "visit", "uct"], default is cumulated utility.
        by selection sort, the swaps during sort is collected as preference pairs.
        return the preference pairs
        :param threshold: handcrafted threshold for preferred
        """
        item_list = mc_tree.layer_traversal(value_type=value_type)
        root_data, root_value=item_list.pop(0)  # the root is valued
        desired_language = self.supported_langs.get_lang(mc_tree.root.children[0].state["lang_code"])
        cleaned_dict = OrderedDict()  # clear redundancy with mean values for each tree nodes
        for item_data, item_value in item_list:  # check the item data's language code:
            detect_language=self.detect_lang(item_data)
            root_language = self.detect_lang(root_data)
            if root_language!="Null_language" and detect_language!=desired_language:
                item_value = item_value/2
            if item_data not in cleaned_dict:
                cleaned_dict[item_data] = [item_value]
            else:
                cleaned_dict[item_data].append(item_value)
        cleaned_list = [(item_data, np.array(item_value).sum()) for item_data, item_value in cleaned_dict.items()]

        chosen = []
        rejected = []
        src_lang_codes = []
        trg_lang_codes = []
        prompts = []
        winrates = []  # record the winrate of the prefered over rejected
        if threshold==0.0:
            threshold = root_value  # default threshold by root value.
        src_lang_code=mc_tree.root.state["lang_code"]
        trg_lang_code=mc_tree.root.children[0].state["lang_code"]
        for i in range(len(cleaned_list)):  # select-sort for preference pairs
            item_i, value_i = cleaned_list[i]
            for j in range(i+1, len(cleaned_list)):
                item_j, value_j = cleaned_list[j]
                if value_j>value_i and value_j>threshold:  # needs swap --> a preference pair
                    # exclude the erroreous lang_code preference.
                    if self.detect_lang(item_j) == desired_language and "[COT]" not in item_j:
                        chosen.append(item_j)
                        # rejection = aggregate_rejection(item_i)
                        rejection = item_i
                        if len(rejection)==0:
                            print(">>> warning: empty line as rejection")
                        rejected.append(rejection)  # rejected.append(aggregate_rejection(item_i))
                        prompts.append(root_data)
                        src_lang_codes.append(src_lang_code)
                        trg_lang_codes.append(trg_lang_code)
                        winrates.append(
                            np.exp(value_j)/( np.exp(value_j)+ np.exp(value_i))
                        )
                        break
                    # swap the data
                    cleaned_list[i], cleaned_list[j] = cleaned_list[j], cleaned_list[i]
        out_data = {}
        out_data["prompt"] = prompts
        out_data["src_lang_code"] = src_lang_codes
        out_data["trg_lang_code"] = trg_lang_codes
        out_data["chosen"] = chosen
        out_data["rejected"] = rejected
        out_data["winrates"] = winrates
        out_df = DataFrame(out_data)
        out_df = out_df.drop_duplicates()
        out_df.reset_index(drop=True, inplace=True)
        return out_df

    def simulate(self, llm, trg2value:str, origin_src:str,
            src_lang_code:str, trg_lang_code:str,
            max_simulation_depth:int=2, semantic_threshod:float=0.25
        )->List:
        """
        greedy translate and back-translate to origin_src until certain depth, reconstruction decay
        below a threshold is deprecated to 0.
        simulation will involve reconstruction circuits through sampled self-play languages {z_i}.
        "y -> z_i -> x"
        simulation doesn't involves contexted exploration with fixed translation prompt.
        reconstruction value by refering to origin_src if provided. origin_src and src_list are in same shape

        :param llm: an llm object
        :param trg2value: a sentence to simulate
        :param origin_src: a sentence to compare with the simulations
        :param src_lang_code: the language code of origin_src
        :param trg_lang_code: the language code of trg2value
        return the best reconstruction with overall meaned bleurt value (same size with trgs2value)
        """
        src_lang = self.supported_langs.get_lang(src_lang_code)
        trg_lang = self.supported_langs.get_lang(trg_lang_code)

        simulated_inputs = [trg2value]  # starts with the trg2value as list
        simulated_inputs_lang = [trg_lang]
        with torch.no_grad():
            translate_prompt = TRANS_PROMPTS[0]
            # direct reconstruction of a dummy src
            temp=translate_prompt.format(src_lan=trg_lang,trg_lan=src_lang,src_sent=trg2value)+LABEL_MARK
            dummy_src = self.default_inference(llm, [temp], sample_mode=False)[0][0]

            for _ in range(max_simulation_depth):  # simulation rounds
                # sample bridge languages
                bridge_lang_code = random.choices(
                    [i for i in self.self_play_lang_codes if i!=src_lang_code and i!=trg_lang_code],
                    k=self.sample_size
                )
                bridge_lang = [self.supported_langs.get_lang(code) for code in bridge_lang_code]

                src_inputs = []
                for line, in_lang in zip(simulated_inputs, simulated_inputs_lang):
                    for z_lang in bridge_lang:
                        temp_line = translate_prompt.format(src_lan=in_lang, trg_lan=z_lang, src_sent=line)
                        src_inputs.append(temp_line +LABEL_MARK)
                bridge_trans = self.default_inference(llm, src_inputs, sample_mode=False)[0]  # sents * sample_size
                # print("bridges:", bridge_trans)
                recons_inputs = []
                for z_lang, bridge_trans in zip(bridge_lang*len(simulated_inputs), bridge_trans): # sents * sample_size
                    temp_line = translate_prompt.format(src_lan=z_lang, trg_lan=src_lang, src_sent=bridge_trans)
                    recons_inputs.append(temp_line +LABEL_MARK)
                recon_list = self.default_inference(llm, recons_inputs, sample_mode=False)[0]
                simulated_inputs = recon_list
                simulated_inputs_lang = [src_lang]*len(simulated_inputs)

            # reciprocal reconstruction above threshold (early-death) as simulated rewards
            recon_value = np.array(self.score(references=[origin_src]*len(simulated_inputs), candidates=simulated_inputs))
            recon_value_dummy = np.array(self.score(references=[dummy_src]*len(simulated_inputs), candidates=simulated_inputs))
            recon_value[recon_value<semantic_threshod] = 0.  # straightforward translation
            recon_value_dummy[recon_value_dummy<semantic_threshod] = 0.  # explainative translation

            best_recon = simulated_inputs[(recon_value_dummy+recon_value).argmax()]  #
            mean_value = max(np.mean(recon_value), np.mean(recon_value_dummy))
            # print(">>> simulate:", mean_value, " ==> ",best_recon)
        return best_recon, mean_value


    def step_explore(
            self, llm, src_sent:str,
            src_lang_code:str, trg_lang_code:str,
            trans_context:dict=None, sample_mode:bool=True
        )->List:
        """
        explore one-step translations for a **single** sequence (via random prompt)

        return the translation process specified by src_lang_code and trg_lang_code in the following rounds
        :param llm: the serving vllm instance
        :param src_sent: a single sentence
        :param trans_context: a list of dict object (at least 2 for current language pair) = [{src_lang_code:"", trg_lang_code:""}]
        :param sample_mode: if not return the best one with max score.
        :return: a list of lists translations [sentence_index * sample_size] with generation scores
        """
        # process the input for agent (LLM)'s translation
        src_lang = self.supported_langs.get_lang(src_lang_code)
        trg_lang = self.supported_langs.get_lang(trg_lang_code)
        translate_prompt = random.choice(TRANS_PROMPTS)
        trans_input = translate_prompt.format(
            src_lan=src_lang, trg_lan=trg_lang, src_sent=src_sent)
        if trans_context is not None:  # merging the exploration via context
            assert len(trans_context)==2, "trans_context must be two dicts of language pairs"
            translate_prompt = random.choice(TRANS_PROMPTS)
            contexted_input = translate_prompt.format(src_lan=src_lang, trg_lan=trg_lang, src_sent=src_sent)
            for item in trans_context:  # traverse the context
                context_prompt = random.choice(TRANS_CONTEXT_PROMPT)
                context = context_prompt.format(
                    src_lan=src_lang, word_pair_src=item[src_lang_code],
                    trg_lan=trg_lang, word_pair_trg=item[trg_lang_code])
                contexted_input = context + contexted_input
            input = [trans_input, contexted_input]
        elif trans_context is None and sample_mode: # fast initiation without context, explore with input perturbation with whitespace (semantic reserving).
            # translate_prompt = random.choice(TRANS_PROMPT)
            # num_white_spaces = len(src_sent)//20
            # perturbed_src_l = list(src_sent)
            # if num_white_spaces >0:
            #     positions = random.sample(range(len(src_sent)), num_white_spaces)
            #     for pos in positions:
            #         perturbed_src_l[pos] += " "
            # perturbed_src_sent = "".join(perturbed_src_l)
            # perturbed_input = translate_prompt.replace("<src_lan>", src_lang).replace("<trg_lan>", trg_lang).replace("<src_sent>", perturbed_src_sent)+LABEL_MARK
            input = [trans_input]
        else:  # no merging also no sample
            input = [trans_input]
        # translate:
        trans_list, score_list = self.default_inference(llm=llm, inputs_list=input, flatten=True, sample_mode=sample_mode)
        # if trans_context is not None:
        #     print(">>>> merging (norm, ctxed):", trans_list)
        if not sample_mode:  # return the only the best generation
            best_index = np.array(score_list).argmax()
            return [trans_list[best_index]], [score_list[best_index]]
        else:  # return all results if sample mode
            return trans_list, score_list

    def default_inference(self, llm, inputs_list, flatten=True, sample_mode=True) -> List:
        """
        inference by transformers.generation for a single sentence (with different trans_prompt)
        inputs list may consists of contexted_trans_prompt or trans_prompt
        return outputs list of lists (sentence_index, sample_size) or a list of flattened results
        """
        llm.eval()
        with torch.no_grad():
            if self.tokenizer.chat_template is not None:
                formated_inputs_list = [self.tokenizer.apply_chat_template(
                    make_mt_instruction(l, llm_path=self.tokenizer.name_or_path),
                    tokenize=False, add_generation_prompt=True) for l in inputs_list]
                model_inputs = self.tokenizer(formated_inputs_list, return_tensors="pt", padding=True).to(llm.device)
            else:
                if "alma" in self.tokenizer.name_or_path.lower():
                    pass 
                elif "tower" in self.tokenizer.name_or_path.lower():
                    pass
                else:
                    inputs_list = [l+LABEL_MARK for l in inputs_list] 
                model_inputs = self.tokenizer(inputs_list, return_tensors="pt", padding=True).to(llm.device)
            if sample_mode:
                generation_out = llm.generate(
                    **model_inputs, generation_config=self.sample_config,
                    return_dict_in_generate=True, output_scores=True)
            else:
                generation_out = llm.generate(
                    **model_inputs, generation_config=self.generate_config,
                    return_dict_in_generate=True, output_scores=True)
            output_seq = generation_out.sequences.reshape(model_inputs["input_ids"].shape[0],self.sample_size if sample_mode else 1, -1)
            input_length = model_inputs["input_ids"].shape[1]
            generated_seqs = output_seq[:,:, input_length:]
            transition_scores = llm.compute_transition_scores(
                generation_out.sequences, generation_out.scores,
                normalize_logits=True
            )  # batch_size, sample_size, gen_length
            length_penalty = llm.generation_config.length_penalty
            real_gen_len = (~transition_scores.isinf()).sum(dim=-1)
            transition_scores[transition_scores.isinf()]=0
            scores = transition_scores.sum(dim=-1) / (real_gen_len ** length_penalty)
            final_results = []
            for out_l in generated_seqs:
                decoded_results = self.tokenizer.batch_decode(out_l, skip_special_tokens=True)
                if flatten:
                    final_results.extend(decoded_results)   # flattened results
                else:
                    final_results.append(decoded_results)
        return final_results, torch.exp(scores).cpu().numpy().tolist()

    def update_policy(self, tuning_dataset:Dataset):
        # set the policy to train mdoe, deploy the preference trainer for epoch update over collected data.
        self.model.is_parallelizable=True
        self.model.model_parallel=True
        print(f">>> rl tuning at lr: {self.pref_train_config.learning_rate}...")
        rl_trainer = DPOTrainer(
            self.model,
            self.base,
            args=self.pref_train_config,
            train_dataset=tuning_dataset,
            tokenizer=self.tokenizer,
            force_use_ref_model=True
        )
        train_results = rl_trainer.train(
            # resume_from_checkpoint=True if os.path.exists(os.path.join(self.agent_out_dir, "trainer_state.json")) else None
        )
        metrics = train_results.metrics
        rl_trainer.log_metrics("train", metrics)
        rl_trainer.save_metrics("train", metrics)
        rl_trainer.save_state()
        loss = rl_trainer.state.log_history[-1]["train_loss"]
        if dist.get_rank()==0:
            if self.args.use_lora:
                rl_trainer.save_model(output_dir=os.path.join(self.agent_out_dir, "rl_adaptor"))
                self.model = self.model.merge_and_unload()
            self.model.save_pretrained(self.agent_out_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(self.agent_out_dir)
        rl_trainer.accelerator.free_memory() # memory leak: release the gpu by accelerator!
        del rl_trainer
        free_gpu()
        print("finish tuning epoch")
        return loss

    def distributed_valued_by_mcts(self, inputs_list, src_lang_code, trg_lang_code):
        sampler  = torch.utils.data.distributed.DistributedSampler(inputs_list)
        # each mcts is distributed with random target language code
        dataloader = DataLoader(
            inputs_list, batch_size=1, sampler=sampler
        )
        dist_results = []
        for _, line in enumerate(dataloader):
            pass
            mc_results = self.value_by_MCTS(
                src_sent=line[0],
                src_lang_code=src_lang_code, trg_lang_code=trg_lang_code,
                max_simulation_depth=4
            )
            mc_results["input"] = line[0]
            mc_results["src_lang_code"] = src_lang_code
            mc_results["trg_lang_code"] = trg_lang_code
            print_once(mc_results)
            dist_results.append(mc_results)
        # yield to the distributed cache
        MC_df = pd.DataFrame(dist_results) # data frame for RL tuning
        MC_df.to_csv(
            os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+f".{dist.get_rank()}"),
            index=False
        )
        free_gpu()
        dist.barrier()

        if dist.get_rank()==0:
            collect_df = []
            cache_paths = glob.glob(os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+f".*"))
            for res_path in cache_paths:
                distributed_df = pd.read_csv(res_path)
                dict4CPO = gen_rank_pair(distributed_df)
                for i in range(len(dict4CPO)):  # update the prompts
                    translate_prompt = random.choice(TRANS_PROMPTS)
                    in_line = dict4CPO.at[i, 'prompt']
                    src_code = dict4CPO.at[i, 'src_lang_code']
                    trg_code = dict4CPO.at[i, 'trg_lang_code']
                    dict4CPO.at[i, 'prompt'] = translate_prompt.format(
                        src_lan=self.supported_langs.get_lang(src_code),
                        trg_lan=self.supported_langs.get_lang(trg_code),
                        src_sent=in_line
                    )
                collect_df.append(dict4CPO)
            # print(collect_df)
            merged_df = pd.concat(collect_df, ignore_index=True)
            merged_df.to_csv(
                os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv"),
                index=False
            )
        return os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv")
