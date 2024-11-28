# -*- coding: utf-8 -*-
import os, json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import random

from datasets import load_dataset
from configs.prompts import TRANS_PROMPT
from configs.lang_codes import ISO2wFamily_ISO2codes
from bleurt_pytorch import BleurtTokenizer, BleurtForSequenceClassification

base_data_path = "/mnt/bn/v2024/dataset/nist_zh-en/"

def generate_alpaca_data(data_path, max_sample=-1):
    """
    return the json output file path, with src2trg and trg2src translation data
    """
    src_lang="Chinese"
    trg_lang="English"

    with open(os.path.join(data_path,"train.zh"), "r") as src_lines, \
        open(os.path.join(data_path, "train.en"), "r") as trg_lines, \
        open(os.path.join(data_path, "nist_zh-en.json"),"w") as dataset_file:
        dataset = []
        sample_count = 0
        for s_l, t_l in zip(src_lines, trg_lines):
            if max_sample>0 and sample_count>max_sample:
                break
            if random.uniform(0,1)>0.5:  # too large data will result in parquet error
                dataset.append({
                    "instruction": TRANS_PROMPT.replace("<src_lan>", src_lang).replace("<trg_lan>", trg_lang),
                    "input": s_l.replace(" ", "").strip(),
                    "output": t_l.strip()
                })
            else:
                dataset.append({
                    "instruction": TRANS_PROMPT.replace("<src_lan>", trg_lang).replace("<trg_lan>", src_lang),
                    "input": t_l.strip(),
                    "output": s_l.replace(" ", "").strip()
                })
            sample_count+=1
        print(">>>>",sample_count)
        random.shuffle(dataset)
        json.dump(dataset, dataset_file, ensure_ascii=False, indent=2)
    return os.path.join(data_path, "nist_zh-en.json")

def generate_alpaca_test_data(data_path):
    src_lang="Chinese"
    trg_lang="English"
    with open(data_path, "r") as src_lines, \
      open(data_path+".json", "w") as out_file:
        data=[]
        sample_count = 0
        for s_l in src_lines:
            data.append({
                "instruction": TRANS_PROMPT.replace("<src_lan>", src_lang).replace("<trg_lan>", trg_lang),
                "input": s_l.replace(" ", "").strip(),
            })
            sample_count+=1
        print(">>>>", sample_count)
        json.dump(data, out_file, ensure_ascii=False, indent=2)
    return os.path.join(base_data_path, data_path+".json")

def generate_parquet_data(src_file, src_lan_code, trg_file, trg_lan_code, output_file):  
    # read the parallel files
    # extract the schema from a parquet file
    # yield according to schema
    with open(src_file, "r") as src_lines, open(trg_file, "r") as trg_lines:
        full_data = []
        for src_l, trg_l in zip(src_lines, trg_lines):
            full_data.append({src_lan_code:src_l.strip(), trg_lan_code:trg_l.strip()})
        df = pd.DataFrame({"translation": full_data})
        df.to_parquet(output_file, index=False)
        # batch = pa.RecordBatch.from_arrays(
        #     [src_array, trg_array],
        #     schema=array_schema
        # )
        # out_table=pa.Table.from_batches([batch])
        # pq.write_table(out_table, output_file)
    return

def process_flores_data(flores_script, output_file, sample_size=-1):
    """
    process the multilingual parallel data to parallel lines.
    exclude some language pairs during translation
    :param flores_script: a .py file that defines flores GeneratorBasedBuilder 
    :param output_file: for the processed parallel in json 
    :param sample_size: the size of the dataset, -1 for full data.
    flores200 adopts the ISO2 language codes
    run by: process_flores_data("/mnt/bn/v2024/dataset/flores200_dataset/flores.py", output_file="flores200.parquet")

    """
    # lang_codes = ["eng", "fra","zho_simpl", "deu", "rus", "kor", "jpn", "ara", "heb", "swh" ] # , 
    lang_codes = ["eng_Latn", "fra_Latn", "zho_Hans", "deu_Latn", "rus_Cyrl", "kor_Hang", "jpn_Jpan", "arb_Arab", "heb_Hebr", "swh_Latn"]
    # flores_data = load_dataset(flores_script, "all")["dev"]  # load all languages
    clense_pair = ["eng_Latn", "zho_Hans"]
    # clense_pair = []
    full_data = []
    for i in range(len(lang_codes)-1):
        src_lan_code = lang_codes[i]
        src_col =load_dataset(flores_script, src_lan_code, trust_remote_code=True)["dev"]["sentence"]
        for j in range(i+1, len(lang_codes)):
            trg_lan_code = lang_codes[j]
            if src_lan_code in clense_pair and trg_lan_code in clense_pair:
                print(f"---skip--- {src_lan_code} -> {trg_lan_code}")
                continue
            else:
                print(f"collect {src_lan_code} -> {trg_lan_code}")
                trg_col = load_dataset(flores_script, trg_lan_code, trust_remote_code=True)["dev"]["sentence"]
                pairs_count = len(lang_codes)*(len(lang_codes)-1)//2
                size4pair = max(1, sample_size//pairs_count) if sample_size>0 else len(src_col)
                total_pairs = [(src,trg) for src,trg in zip(src_col, trg_col) ]
                random_pairs = random.sample(total_pairs, size4pair)
                for src_l, trg_l in random_pairs:
                    full_data.append({src_lan_code:src_l.strip(), trg_lan_code:trg_l.strip()})
    print(">> total >>", len(full_data))
    df = pd.DataFrame({"translation": full_data})
    df.to_json(output_file, orient='records', lines=True)
    return 

def process_flores_instruct(flores_script, output_file, sample_size=-1):
    all_data= load_dataset(flores_script, "all", trust_remote_code=True)["dev"]
    flores_colums = all_data.column_names
    supported_lang_code = list(ISO2wFamily_ISO2codes.keys())
    data_size = len(all_data["sentence_eng_Latn"]) 

    full_data = []
    data_count=0
    for i in range(len(supported_lang_code)-1):
        src_lang_code = supported_lang_code[i]
        if f"sentence_{src_lang_code}" in flores_colums:
            for j in range(i+1, len(supported_lang_code)):
                trg_lang_code=supported_lang_code[j]
                if f"sentence_{trg_lang_code}" in flores_colums:
                    sample_index = random.randint(0, data_size-1)
                    src_l = all_data[f"sentence_{src_lang_code}"][sample_index]
                    trg_l = all_data[f"sentence_{trg_lang_code}"][sample_index]
                    full_data.append({src_lang_code: src_l.strip(), trg_lang_code: trg_l.strip()})
                    data_count+=1
                    if sample_size>0 and data_count>sample_size:
                            break
    print(">> total >>", len(full_data))
    for i in range(len(full_data)):
        if len(full_data[i].keys())!=2:
            print("error")
    df = pd.DataFrame({"translation": full_data})
    table = pa.Table.from_pandas(df)
    pq.write_table(table, output_file)
    # df.to_parquet(output_file, index=False)              
    return
    
def process_flores_test(flores_script, src_lang_code, trg_lang_code, output_file="flores_test"):
    """
    extract the flores200 test data to parallel lines.
    df title: translation
    {src_lang_code: src_lang_sent, trg_lang_code: trg_lang_sent}

    flores200 adopts the ISO2 language codes
        e.g., ["eng_Latn", "fra_Latn", "zho_Hans", "deu_Latn", "rus_Cyrl", "kor_Hang", "jpn_Jpan", "arb_Arab", "heb_Hebr", "swh_Latn"]

    run by: process_flores_test("/mnt/bn/v2024/dataset/flores200_dataset/flores.py", "eng_Latn","zho_Hans")

    pair_code: with dash line '-' 
    """
    # lang_codes = ["eng", "fra","zho_simpl", "deu", "rus", "kor", "jpn", "ara", "heb", "swh" ] # ,
    print(f"collect flores test on {src_lang_code} with {trg_lang_code}...")
    para_data = []
    lan_pair = load_dataset(flores_script, f"{src_lang_code}-{trg_lang_code}", trust_remote_code=True )["devtest"]
    for i in range(len(lan_pair)):
        para_data.append(
            {src_lang_code:lan_pair[i][f"sentence_{src_lang_code}"], trg_lang_code: lan_pair[i][f"sentence_{trg_lang_code}"]}
        )
    df = pd.DataFrame({"translation": para_data})
    df.to_parquet(f"{output_file}_{src_lang_code}-{trg_lang_code}.parquet", index=False)
    print(f"finsh at {output_file}_{src_lang_code}-{trg_lang_code}.parquet")
    return

def sample_parquet_data(parquet_file, sample_size):
    df = pd.read_parquet(parquet_file)
    sampled_df = df.sample(sample_size)
    sampled_df.to_parquet(parquet_file.replace(".parquet", ".%d.parquet"%sample_size))
    return

# generate_parquet_data(
#     "/mnt/bn/v2024/dataset/nist_zh-en/train.zh", "zh",
#     "/mnt/bn/v2024/dataset/nist_zh-en/train.en", "en",
#     "/mnt/bn/v2024/dataset/nist_zh-en/nist_zh-en.parquet",
# )
# alpaca_data_path=generate_alpaca_data("/mnt/bn/v2024/dataset/nist_zh-en/")
# alpaca_test_path = generate_alpaca_test_data("/mnt/bn/v2024/dataset/nist_zh-en/test/mt08.src")
# process_flores_data("/mnt/bn/v2024/dataset/flores200_dataset/flores.py", output_file="flores200.parquet", sample_size=-1)
process_flores_instruct("/mnt/bn/v2024/dataset/flores200_dataset/flores.py", output_file="flores200.parquet")
# process_flores_test("/mnt/bn/v2024/dataset/flores200_dataset/flores.py", "zho_Hans","arb_Arab")

def calculate_bleurt(ref_list, cand_list):
    bleurt_path="/mnt/bn/v2024/models/huggingface/bleurt20/"
    score_tokenizer = BleurtTokenizer.from_pretrained(bleurt_path)
    scorer = BleurtForSequenceClassification.from_pretrained(bleurt_path)
    scorer.eval()
    inputs = score_tokenizer(ref_list, cand_list, padding="longest", return_tensors="pt")
    score = scorer(**inputs).logits.flatten().mean().tolist()
    return score

def merge_mono_data(dir="/mnt/bn/v2024/dataset/monolingual/rus_Cyrl"):
    with open(os.path.join(dir, "news.shuffled"), "r") as in_file, \
     open(os.path.join(dir,"merged.txt"),"w") as out_file:
        news_lines = []
        for l in in_file:
            if 20<len(l) and len(l)<200:
                news_lines.append(l.strip())
        print(len(news_lines))
        # literature = pd.read_csv(os.path.join(dir, "literature.csv"))
        literature = pd.read_parquet(os.path.join(dir, "literature.parquet"))
        # literature_lines = literature["text"].tolist()
        literature_lines = literature["text"].tolist()
        literature_lines = [l.strip().replace("\n"," ") for l in literature_lines if 20<len(l)<200]
        news_lines.extend(literature_lines)

        random.shuffle(news_lines)
        for l in news_lines:
            out_file.write(l+"\n")

# merge_mono_data("/mnt/bn/v2024/dataset/monolingual/zho_Hans")