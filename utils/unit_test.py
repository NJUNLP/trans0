from modules.agent import TransAgent
import time, os, glob
import json
from collections import OrderedDict
import numpy as np
import pyarrow as pa
from modules.RewardModel import RewardModel, reward_model_dir
from modules.data import read_parallel_data
from datasets import Dataset
import pandas as pd
import torch.distributed as dist
from utils.common_utils import free_gpu
from configs.prompts import RM_PROMPT
from configs.lang_codes import LangCodes

def RL_update(args, train_round:int):
    agent = TransAgent(args, train=train_round)
    start = time.time()
    cached_files = glob.glob(os.path.join(agent.cache_dir, f"self_play_*.csv"))
    cached_SP_dir = sorted(cached_files, key=lambda x: os.path.getmtime(x), reverse=True)[0]
    merged_df = pd.read_csv(cached_SP_dir)
    tuning_dataset = Dataset(pa.Table.from_pandas(merged_df))
    print("loading RL finetune data.")
    agent.update_policy(tuning_dataset) 
    end = time.time()

    del agent, tuning_dataset, merged_df
    free_gpu()
    print(">>lapse>>:", end-start)
    dist.barrier()

def unit_test(args):
    """
    input the parallel data, calculate the MC values, with generation scores
    yidl to parquet lists 
    """
    # src_list, trg_list = read_parallel_data(args.dev_data_path, src_lang_code = "zh", trg_lang_code = "en")
    input_line = "If you always harm others, the chickens' gonna come home to roost."  # eng_Latn
    # input_line = "Putins Verrücktes Spiel"  # deu_Latn
    # input_line="التحدي الجديد لمنتدى دافوس" # arb_Arab
    # input_line = "我世祖文皇帝，神文圣武，继承大统，应天合人，法尧禅舜，处中国以治万 邦，这岂非天心人意乎？"  # zho_Hans
    # input_line="Syriza stand einst für eine Abkehr vom Euro"  # deu_Latn
    agent = TransAgent(args)
    mc_tree = agent.MCTS(
        src_sent=input_line, 
        src_lang_code="eng_Latn", trg_lang_code="zho_Hans", max_simulation_depth=2
    )
    item_list = mc_tree.layer_traversal(value_type="utility")
    root_data, root_value = item_list.pop(0)
    print(f"{root_value}:{root_data}")
    cleaned_dict = OrderedDict()
    for item_data, item_value in item_list:
        if item_data not in cleaned_dict:
            cleaned_dict[item_data] = [item_value]
        else:
            cleaned_dict[item_data].append(item_value)
    cleaned_list = [(item_data, (np.array(item_value)).sum()) for item_data, item_value in cleaned_dict.items()]
    for item_data, item_value in cleaned_list:
        print(f"{item_value}:{item_data}")

    df=agent.yield_tree2rank(mc_tree, value_type="utility")
    df.to_csv("preference.csv", index=False)

    # for i in range(3):
    #     RL_update(args, train_round=i)
    #     input()


    # df = pd.DataFrame(results)
    # df.to_csv(args.dev_data_path.split("/")[-1]+".log", index=False)
    return 

def validate_preference(self_play_data:str):   # **-**.*.self_play*.csv  
    df = pd.read_csv(self_play_data).dropna()
    rm_model = RewardModel(reward_model_dir)
    supported_langs = LangCodes()
    count = 0
    for i in range(len(df)):
        src_lang_code = df.iloc[i]["src_lang_code"]
        trg_lang_code = df.iloc[i]["trg_lang_code"]
        chosen_seq = df.iloc[i]["chosen"]
        reject_seq = df.iloc[i]["rejected"]
        input_line = df.iloc[i]["prompt"]
        
        query = RM_PROMPT.replace("<src_lan>", supported_langs.get_lang(src_lang_code)).\
            replace("<trg_lan>", supported_langs.get_lang(trg_lang_code)).replace("<src_sent>", input_line)

        score = rm_model.score(prompts=[query]*2, chosens=[chosen_seq, reject_seq])
        if score[0] > score[1]:
            count += 1
        print(f"score:{score}")
    print(count, len(df), float(count)/len(df))
    return float(count)/len(df)

# validate_preference("/mnt/bn/v2024/cache/llama3-mega_clax/trans0_agent/zho_Hans-eng_Latn.18.self_play_68.csv")
