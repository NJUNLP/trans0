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

