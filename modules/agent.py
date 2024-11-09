# -*- coding: utf-8 -*-
import glob
import os
import random
from typing import List, OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer
from datasets import Dataset
from pandas import DataFrame
from peft import PeftConfig, PeftModel, get_peft_model
from sacrebleu.metrics import BLEU, CHRF
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from trl import CPOConfig, CPOTrainer, DPOConfig, DPOTrainer

from configs.configs import sp_peft_config
from configs.lang_codes import LangCodes
from configs.prompts import LABEL_MARK, TRANS_CONTEXT_PROMPT, TRANS_PROMPT
from modules.data import gen_rank_pair
from modules.NaryTree import *
from utils.common_utils import (
    aggregate_rejection,
    free_gpu,
    get_path,
    print_once,
    truncate_encoded,
)

# from torch.nn.parallel import DistributedDataParallel as DDP

# from sacrebleu.metrics import BLEU

class TransAgent:
    # wrap the methods and functions for trans0.
    def __init__(self, args, train=None, override_cache=False, metric_type="bleurt"):
        # initiate agent by SFTed LLM for translation.
        self.args = args  # reserve the parameters
        self.sample_size = args.mcts_sample_size if args.mcts_sample_size else 4
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

        self.tokenizer = AutoTokenizer.from_pretrained(
            ckpt_path,
            model_max_length=args.max_length,  # controls the maximum PE
            padding_side = args.padding_side,
            truncation_size = args.truncation_side,
            trust_remote_code=True
        )
        self.base = None  # base is the ref model in DPO training and BT evaluation in MCTS simulation.
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
                    get_path(args, args.output_dir),
                    load_in_8bit=False,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
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
                    get_path(args, args.output_dir),
                    load_in_8bit=False,
                    trust_remote_code=True,
                    torch_dtype=torch.bfloat16,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    ckpt_path, trust_remote_code=True #, device_map=f"cuda:{dist.get_rank()}"
                    ).to("cuda")
                self.base=self.model
                # self.base = AutoModelForCausalLM.from_pretrained(
                #     args.output_dir, trust_remote_code=True, device_map=f"cuda:{dist.get_rank()}")
        self.model.is_parallelizable=True
        self.model.model_parallel=True

        self.metric_type = metric_type
        if self.metric_type=="chrf":
            self.scorer = CHRF()
        elif self.metric_type=="bleu":
            self.scorer = BLEU()
        elif self.metric_type=="bleurt":
            self.scorer_tokenizer = BleurtTokenizer.from_pretrained(get_path(args,args.bleurt_ckpt))
            self.scorer = BleurtForSequenceClassification.from_pretrained(get_path(args,args.bleurt_ckpt)).to("cuda")
            self.scorer.eval()
        else:
            raise ValueError("Invalid metric_type. Please choose from 'chrf', 'bleu', or 'bleurt'.")

        self.generate_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=1,
        )  # for generation
        self.sample_config = GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=0.5,
            do_sample=True,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_return_sequences=self.sample_size,
        )  # for sampling
        self.pref_train_config = DPOConfig(
            logging_steps=args.logging_steps,
            max_length=args.max_length,
            max_prompt_length=1024,
            output_dir=self.agent_out_dir,
            learning_rate=args.rl_learning_rate,
            lr_scheduler_type=args.rl_lr_scheduler_type,
            loss_type=args.rl_loss_type,  # cpo_alpha=.0,
            per_device_train_batch_size=1,
            deepspeed=args.deepspeed,
            gradient_accumulation_steps=4
            * args.gradient_accumulation_steps
            * args.per_device_train_batch_size,
            resume_from_checkpoint=True,
            save_strategy="no",
            remove_unused_columns=args.remove_unused_columns,
            bf16=args.bf16,
            tf32=args.tf32,
        )

        self.supported_langs = LangCodes()

    def score(self, references:List[str], candidates:List[str])->List:
        """
        wraping the score interface to evaluate reconstructions return a list of scores
        """
        if self.metric_type =="bleurt":
            with torch.no_grad():
                # one-on-one bleurt with a list of score output
                inputs1 = self.scorer_tokenizer(references, candidates, padding="longest", return_tensors='pt').to(self.scorer.device)
                trunc_input1 = truncate_encoded(inputs1)
                res1 = self.scorer(**trunc_input1).logits.flatten()
                inputs2 = self.scorer_tokenizer(candidates, references, padding="longest", return_tensors="pt").to(self.scorer.device)
                trunc_input2 = truncate_encoded(inputs2)
                res2 = self.scorer(**trunc_input2).logits.flatten()
                res=((res1+res2)/2).tolist()
                return res
        elif self.metric_type in ["chrf", "bleu"]:
            scores = []
            for ref,cand in zip(references, candidates):
                score1=self.scorer.sentence_score(hypothesis=cand, references=[ref]).score
                score2=self.scorer.sentence_score(hypothesis=ref, references=[cand]).score
                scores.append((score2 + score1) / 2)
            return scores

    def valued_by_BLEUrt(
        self, inputs_list, targets_list, src_lang_code: str, trg_lang_code: str
    ):
        """
        used for RL hyperparameter finetuning.
        """
        llm = self.model
        llm.eval()
        if dist.get_rank()==0:
            with torch.no_grad():
                mc_results = []
                for src_line, trg_line in zip(inputs_list, targets_list):
                    explored_trgs, scores = self.step_explore(
                        llm, src_line, src_lang_code=src_lang_code, trg_lang_code=trg_lang_code, sample_mode=False
                    )
                    # calculate the reconstructed trgs with reference trgs for bleurt
                    rewards = self.score(references=[trg_line]*len(explored_trgs), candidates=explored_trgs)
                    collected = {"input": src_line, "src_lang_code": src_lang_code, "trg_lang_code": trg_lang_code,
                        "sequences":explored_trgs, "scores":scores, "values":rewards}
                    print(collected)
                    mc_results.append(collected)
                free_gpu()
                # yield to the distributed cache
                MC_df = pd.DataFrame(mc_results) # data frame for RL tuning
                MC_df.to_csv(
                    os.path.join(
                        self.cache_dir,
                        self.args.dev_data_path.split("/")[-1] + f".{dist.get_rank()}",
                    ),
                    index=False,
                )
                collect_df = pd.read_csv(os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+f".{dist.get_rank()}"))
                dict4CPO = gen_rank_pair(collect_df)
                for i in range(len(dict4CPO)):  # update the prompts
                    translate_prompt = random.choice(TRANS_PROMPT)
                    in_line = dict4CPO.at[i, 'prompt']
                    src_code = dict4CPO.at[i, 'src_lang_code']
                    trg_code = dict4CPO.at[i, 'trg_lang_code']
                    dict4CPO.at[i, "prompt"] = (
                        translate_prompt.replace(
                            "<src_lan>", self.supported_langs.get_lang(src_code)
                        )
                        .replace("<trg_lan>", self.supported_langs.get_lang(trg_code))
                        .replace("<src_sent>", in_line)
                    )
                dict4CPO.to_csv(
                    os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv"),
                    index=False
                )
        dist.barrier()
        return os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv")

    def value_by_MCTS(
        self,
        src_sent: str,
        src_lang_code: str,
        trg_lang_code: str,
        max_simulation_depth: int,
    ):
        """
        a dev used for RL tuning. more max_simnulation depth for larger ranking variance.
        """
        llm = self.model
        llm.eval()
        assert self.supported_langs.check_support(src_lang_code), "source must be supported languages"
        assert self.supported_langs.check_support(trg_lang_code), "target must be supported languages"
        # the exploration includes the sampled inference by origin prompts and contexted prompts
        with torch.no_grad():
            explored_trgs, scores = self.step_explore(
                llm,
                src_sent,
                src_lang_code=src_lang_code,
                trg_lang_code=trg_lang_code,
                sample_mode=False,
            )
            # add the explored trgs to tree and back-translation for semantic rewards
            recon_srcs = []
            for t_line in explored_trgs:  # evaluate by base model reconstruction
                recon_srcs.extend(
                    self.step_explore(
                        self.base,
                        t_line,
                        src_lang_code=trg_lang_code,
                        trg_lang_code=src_lang_code,
                        sample_mode=False,
                    )[0]
                )
            rewards_flat = self.score(references=[src_sent]*len(recon_srcs), candidates=recon_srcs)

            recon_src_values = self.simulate(  # simulate with base model
                self.base,
                input_list=recon_srcs,
                src_lang_code=src_lang_code,
                trg_lang_code=trg_lang_code,
                max_simulation_depth=max_simulation_depth,
            )
            rewards_flat = [(a+b*max_simulation_depth)/(max_simulation_depth+1) for (a,b) in zip(rewards_flat, recon_src_values)]
            rewards = np.array(rewards_flat).reshape(-1, self.sample_size).mean(axis=-1).tolist()  # rewards on translation

        return {"sequences": explored_trgs, "scores": scores, "values": rewards}

    def MCTS(
        self,
        src_sent: str,
        src_lang_code: str,
        trg_lang_code: str,
        MC_count: int = 20,
        max_simulation_depth=3,
    ) -> NaryTree:
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
            mc_tree = NaryTree(state={"data":src_sent, "lang_code":src_lang_code, "recon": None})
            root_lang_code = mc_tree.root.state["lang_code"]
            explored_trgs, scores = self.step_explore(
                llm, mc_tree.root.state["data"], src_lang_code=src_lang_code, trg_lang_code=trg_lang_code,
                sample_mode=False
            )
            for t_line in explored_trgs: # a dummy simulation for fast initiation.
                recon_srcs = self.step_explore(
                        self.base, t_line, src_lang_code=trg_lang_code, trg_lang_code=root_lang_code,
                        sample_mode=False
                    )[0]
                rewards = self.score(references=[src_sent]*len(recon_srcs), candidates=recon_srcs)
                best_index = np.array(rewards).argmax(axis=-1)
                child = mc_tree.root.add_child(
                    state={
                        "data": t_line,
                        "lang_code": trg_lang_code,
                        "recon": recon_srcs[best_index],
                    }
                )
                mc_tree.backpropagate(child, value=np.array(rewards).mean()) # updates the child upto the root

            # mc tree search
            best_node_previous = mc_tree.root  # top best nodes for context.
            for count in range(MC_count):
                current_node = mc_tree.select()  # select a leaf for expansion.
                print(f">>>> {count} node:", current_node.state["data"])
                if best_node_previous == current_node or best_node_previous.state["recon"]==None:
                    # expand with Markov translation process
                    # (explored by given back-translated paraphrase, or whitespace paraphrase)
                    explored_trgs, _ = self.step_explore(
                        llm, current_node.state["recon"],
                        src_lang_code=root_lang_code, trg_lang_code=trg_lang_code,
                        trans_context=None
                    )
                else:
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
                recon_srcs, _ = self.step_explore(
                    self.base,
                    explored_trgs[0],
                    src_lang_code=trg_lang_code,
                    trg_lang_code=root_lang_code,
                    sample_mode=False,
                )
                # simulated_value = self.score(references=[src_sent]*len(recon_srcs), candidates=recon_srcs)[0]
                new_node = mc_tree.add_child(
                    parent = current_node,
                    child_data={"data":explored_trgs[0], "lang_code": trg_lang_code, "recon":recon_srcs[0]}
                )
                simulated_value = (
                    self.simulate(  # multi-round simulation is not necessary
                        self.base,
                        input_list=[new_node.state["recon"]],
                        origin_src=[src_sent],
                        src_lang_code=root_lang_code,
                        trg_lang_code=trg_lang_code,
                        max_simulation_depth=max_simulation_depth,
                    )[0]
                )
                # print(">>>> new node", new_node.state["recon"], new_node.state["data"])
                # print(">>>> value", simulated_value)
                mc_tree.backpropagate(new_node, simulated_value)
                # update the node record with best utility
                best_node_previous = mc_tree.get_best(mc_tree.root)
        return mc_tree

    def yield_tree2rank(self, mc_tree:NaryTree, value_type="utility")->DataFrame:
        """
        yield the tree results to a ranking dataframe for training
        a MCT in layerswise traversal example (ancesters are ahead of descendants):
        [('落霞与孤鹜齐飞，秋水共长天一色', 0.529447915361208),   # the root
        ('The setting sun and the lone wild goose fly together, the autumn waters blend with the sky in one color.', 0.5596388263834847),
        ('The setting sun and the lone crane fly together, the autumn water is the same color as the sky.', 0.5597622022032738),
        ('The setting sun and the lone wild goose fly together, the autumn water and the sky are of the same color.', 0.5720633864402771)]

        :param value_type: rank by value: ["utility", "value", "visit", "uct"], default is cumulated utility.
        """
        item_list = mc_tree.layer_traversal(value_type=value_type)
        root_data, root_value = item_list.pop(0)  # the root is valued

        cleaned_dict = OrderedDict()  # clear redundancy with mean values for each tree nodes
        for item_data, item_value in item_list:
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
        for i in range(len(cleaned_list)):
            for j in range(i+1, len(cleaned_list)):
                item_i, value_i = cleaned_list[i]
                item_j, value_j = cleaned_list[j]
                if value_j>value_i:
                    chosen.append(item_j)
                    rejected.append(aggregate_rejection(item_i))  # rejected.append(item_i)
                    prompts.append(root_data)
                    src_lang_codes.append(mc_tree.root.state["lang_code"])
                    trg_lang_codes.append(mc_tree.root.children[0].state["lang_code"])
        out_data = {}
        out_data["prompt"] = prompts
        out_data["src_lang_code"] = src_lang_codes
        out_data["trg_lang_code"] = trg_lang_codes
        out_data["chosen"] = chosen
        out_data["rejected"] = rejected
        out_df = DataFrame(out_data)
        out_df = out_df.drop_duplicates().dropna()
        out_df.reset_index(drop=True, inplace=True)
        return out_df

    def simulate(
        self,
        llm,
        input_list: str,
        src_lang_code: str,
        trg_lang_code: str,
        max_simulation_depth: int,
        semantic_threshod: float = 0.25,
        origin_src: str = None,
    ) -> List:
        """
        greedy translate and back-translate src_sents until certain depth, reconstruction decay
        below a threshold is deprecated to 0.
        simulation doesn't involves contexted exploration
        reconstruction value by refering to origin_src if provided. origin_src and src_list are in same shape
        return mean over all the reconstructed bleurt values
        """
        src_lang = self.supported_langs.get_lang(src_lang_code)
        trg_lang = self.supported_langs.get_lang(trg_lang_code)
        simulated_inputs = input_list
        with torch.no_grad():
            for _ in range(max_simulation_depth):
                translate_prompt= random.choice(TRANS_PROMPT)
                src_inputs = [translate_prompt.replace("<src_lan>", src_lang).replace("<trg_lan>", trg_lang).replace("<src_sent>", item) +LABEL_MARK for item in simulated_inputs]
                _trgs = self.default_inference(llm, src_inputs, sample_mode=False)[0]
                invert_translate_prompt = random.choice(TRANS_PROMPT)
                trg_inputs = [invert_translate_prompt.replace("<trg_lan>", src_lang).replace("<src_lan>", trg_lang).replace("<src_sent>", item) +LABEL_MARK for item in _trgs]
                recon_list = self.default_inference(llm, trg_inputs, sample_mode=False)[0]
                simulated_inputs = recon_list
            # reciprocal reconstruction above threshold (early-death) as simulated rewards
        if origin_src is None:
            recon_value = np.array(self.score(references=input_list, candidates=simulated_inputs))
        else:
            recon_value = np.array(self.score(references=origin_src, candidates=simulated_inputs))
        recon_value[recon_value<semantic_threshod] = 0.
        return recon_value.tolist()

    def step_explore(
        self,
        llm,
        src_sent: str,
        src_lang_code: str,
        trg_lang_code: str,
        trans_context: dict = None,
        sample_mode: bool = True,
    ) -> List:
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
        translate_prompt = random.choice(TRANS_PROMPT)
        trans_input = translate_prompt.replace("<src_lan>", src_lang).replace("<trg_lan>", trg_lang).replace("<src_sent>", src_sent)+LABEL_MARK
        if trans_context is not None:  # merging the exploration via context
            assert len(trans_context)==2, "trans_context must be two dicts of language pairs"
            translate_prompt = random.choice(TRANS_PROMPT)
            for item in trans_context:  # traverse the context
                context_prompt = random.choice(TRANS_CONTEXT_PROMPT)
                context = context_prompt.replace("<word_pair_src>", item[src_lang_code]).replace("<word_pair_trg>", item[trg_lang_code])
                translate_prompt = context + translate_prompt
            contexted_input = translate_prompt.replace("<src_lan>", src_lang).replace("<trg_lan>", trg_lang).replace("<src_sent>", src_sent)+LABEL_MARK
            input = [trans_input, contexted_input]
        elif trans_context is None and sample_mode: # without context, explore with input perturbation with whitespace (semantic reserving).
            translate_prompt = random.choice(TRANS_PROMPT)
            perturbed_src_sent = ""
            for char in src_sent:
                perturbed_src_sent += char+" "
            perturbed_input = translate_prompt.replace("<src_lan>", src_lang).replace("<trg_lan>", trg_lang).replace("<src_sent>", perturbed_src_sent)+LABEL_MARK
            input = [trans_input, perturbed_input]
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
            model_inputs = self.tokenizer(inputs_list, return_tensors="pt", padding=True).to(llm.device)
            if sample_mode:
                generation_out = llm.generate(
                    **model_inputs,
                    generation_config=self.sample_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            else:
                generation_out = llm.generate(
                    **model_inputs,
                    generation_config=self.generate_config,
                    return_dict_in_generate=True,
                    output_scores=True,
                )
            output_seq = generation_out.sequences.reshape(model_inputs["input_ids"].shape[0],self.sample_size if sample_mode else 1, -1)
            input_length = model_inputs["input_ids"].shape[1]
            generated_seqs = output_seq[:,:, input_length:]
            transition_scores = llm.compute_transition_scores(
                generation_out.sequences, generation_out.scores, normalize_logits=True
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
        if dist.get_rank()==0:
            if self.args.use_lora:
                rl_trainer.save_model(output_dir=os.path.join(self.agent_out_dir, "rl_adaptor"))
                self.model = self.model.merge_and_unload()
            self.model.save_pretrained(self.agent_out_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(self.agent_out_dir)
        rl_trainer.accelerator.free_memory()  # memory leak: release the gpu by accelerator!
        del rl_trainer
        free_gpu()
        print("finish tuning epoch")
        return

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
                src_lang_code=src_lang_code,
                trg_lang_code=trg_lang_code,
                max_simulation_depth=4,
            )
            mc_results["input"] = line[0]
            mc_results["src_lang_code"] = src_lang_code
            mc_results["trg_lang_code"] = trg_lang_code
            print_once(mc_results)
            dist_results.append(mc_results)
        # yield to the distributed cache
        MC_df = pd.DataFrame(dist_results) # data frame for RL tuning
        MC_df.to_csv(
            os.path.join(
                self.cache_dir,
                self.args.dev_data_path.split("/")[-1] + f".{dist.get_rank()}",
            ),
            index=False,
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
                    translate_prompt = random.choice(TRANS_PROMPT)
                    in_line = dict4CPO.at[i, 'prompt']
                    src_code = dict4CPO.at[i, 'src_lang_code']
                    trg_code = dict4CPO.at[i, 'trg_lang_code']
                    dict4CPO.at[i, "prompt"] = (
                        translate_prompt.replace(
                            "<src_lan>", self.supported_langs.get_lang(src_code)
                        )
                        .replace("<trg_lan>", self.supported_langs.get_lang(trg_code))
                        .replace("<src_sent>", in_line)
                    )
                collect_df.append(dict4CPO)
            # print(collect_df)
            merged_df = pd.concat(collect_df, ignore_index=True)
            merged_df.to_csv(
                os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv"),
                index=False
            )
        return os.path.join(self.cache_dir, self.args.dev_data_path.split("/")[-1]+".self_play.csv")
