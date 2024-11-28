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
    input_line = "最低的和惟一必要的工资额就是工人在劳动期间的生活费用，再加上使工人能够养家糊口并使工人种族不致死绝的费用。"
    # input_line = "The minimum and only necessary wage is the cost of living for the worker during the working period, plus the expense of allowing the worker to feed his family and not die of starvation."
    # input_line = "적 장사정포, 이동식 발사대 등 지상목표물을 정밀 타격할 수 있는 능력을 갖추고 있는 것으로 평가된다."
    # input_line = "十动然拒，十分感动，然而拒绝。"
    # input_line = "白日依山尽,黄河入海流"
    # input_line = "十动然拒"
    # input_line = "死哪儿风流去了"
    # input_line = "Do not go gentle into that good night, Old age should burn and rave at close of day; Rage, rage against the dying of the light."
    input_line="黄河之水天上来，落霞与孤鹜齐飞，秋水共长天一色"
    agent = TransAgent(args)
    mc_tree = agent.MCTS(
        src_sent=input_line, 
        src_lang_code="zh", trg_lang_code="en", max_simulation_depth=3
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
