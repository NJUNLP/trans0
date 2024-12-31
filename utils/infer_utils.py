from datasets import load_dataset
import pandas as pd

def process_flores_test(flores_script, src_lang_code, trg_lang_code, output_dir):
    """
    extract the flores200 test data of the specific lang-code2lang-code (parallel lines).
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
    df.to_parquet(output_dir, index=False)
    print(f"**** save at {output_dir}")
    return

def process_mix_flores_test(flores_script, self_play_langs, output_dir, per_n_sample=None):
    """
    collect the all the relevant translation pairs (self_play_langs) from flores200
    process_mix_flores_test(flores_script="flores200.py",
            self_play_langs=["eng_Latn", "zho_Hans","deu_Latn", "arb_Arab", "ita_Latn"],
            output_dir="flores200_mix.parquet", per_n_sample=100)

    :param flores_script: the flores.py file
    :param output_dir: dir to save the collect translation pairs.
    :param self_play_langs: a list of language codes to collect.
    :param valid type: the type of parallel pairs.
    """
    all_data= load_dataset(flores_script, "all", trust_remote_code=True)["devtest"]
    flores_colums = all_data.column_names
    data_size = len(all_data["sentence_eng_Latn"])
    if per_n_sample is None:
        per_n_sample = data_size
    full_data=[]
    data_count=0
    for i in range(len(self_play_langs)-1):  # extract the pair combination.
        src_lan_code=self_play_langs[i]
        if f"sentence_{src_lan_code}" in flores_colums:
            for j in range(i+1, len(self_play_langs)):
                trg_lan_code=self_play_langs[j]
                if f"sentence_{trg_lan_code}" in flores_colums:
                    for k in range(per_n_sample):
                        src_l=all_data[f"sentence_{src_lan_code}"][k]
                        trg_l=all_data[f"sentence_{trg_lan_code}"][k]
                        full_data.append({
                            "input_lang_code":src_lan_code, "input":src_l.strip(),
                            "output_lang_code":trg_lan_code, "output":trg_l.strip()})
                        full_data.append({
                            "input_lang_code":trg_lan_code, "input":trg_l.strip(),
                            "output_lang_code":src_lan_code, "output":src_l.strip()})
                        data_count+=2
    print(">> total size:", data_count)
    df = pd.DataFrame({"translation": full_data})
    df.to_parquet(output_dir, index=False)
    return  output_dir

def extract_test(data_path:str, valid_type:str="all"):
    """
    extract the test data from the given data_path.
    :param data_path: the path to the test parquet. ["input_lang_code", "input", "output_lang_code", "output"]
    :param valid_type: the type of parallel pairs. ["en2x", "x2x", "x2en", "all"]
    """
    df = pd.read_parquet(data_path)["translation"]

    if valid_type=="all":
        return df
    elif valid_type=="en2x":
        index = [i for i in range(len(df)) if df.iloc[i]["input_lang_code"]=="eng_Latn"]
        en2x_df = df.iloc[index].reset_index(drop=True)
        print("en2x test sample size: ", len(en2x_df))
        return en2x_df
    elif valid_type=="x2en":
        index = [i for i in range(len(df)) if df.iloc[i]["output_lang_code"]=="eng_Latn"]
        x2en_df = df.iloc[index].reset_index(drop=True)
        print("x2en test sample size: ", len(x2en_df))
        return x2en_df
    elif valid_type=="x2x":
        index = []
        for i in range(len(df)):
            if df.iloc[i]["input_lang_code"]!= "eng_Latn" and df.iloc[i]["output_lang_code"]!= "eng_Latn":
                index.append(i)
        x2x_df = df.iloc[index].reset_index(drop=True)
        print("x2x test sample size: ", len(x2x_df))
        return x2x_df
    else:
        print(">>> invalid valid_type, return None.")
        return None

