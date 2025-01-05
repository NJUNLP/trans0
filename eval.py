import csv
import json
import os
from collections import defaultdict
from pathlib import Path

from modules.metrics import BleurtScorer, CometScorer

LANGS = {
    "zho_Hans": "Chinese",
    "eng_Latn": "English",
    "deu_Latn": "German",
    "fra_Latn": "French",
    "ita_Latn": "Italian",
    "por_Latn": "Portuguese",
    "hin_Deva": "Hindi",
    "spa_Latn": "Spanish",
    "tha_Thai": "Thai",
    "arb_Arab": "Arabic",
    "isl_Latn": "Icelandic",
}

MODELS = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
]


def pred_post_process(pred: str):
    split_indicator = "\n\n"
    if split_indicator in pred:
        pred = pred.split(split_indicator)[1]
    pred = pred.strip()
    return pred


def compute_score(fpath: Path):
    with open(fpath, "r", encoding="utf-8") as f:
        data = json.load(f)
    src_list, ref_list, pred_list = [], [], []
    for d in data:
        src_list.append(d["source"])
        ref_list.append(d["reference"])

        pred = pred_post_process(d["predicted"])
        pred_list.append(pred)

    bleurt_score = bleurt_scorer.score(ref_list, pred_list)
    comet_score = comet_scorer.score(src_list, pred_list)
    return {"BLEURT": bleurt_score, "COMET": comet_score}


def dump_csv_file(data: defaultdict, fpath: Path):
    keys = sorted(data.keys())
    with open(fpath, mode="w+", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["LANG"] + keys)
        for row_key in keys:
            row = [row_key]
            for col_key in keys:
                value = data[row_key].get(col_key, -1)
                row.append(value)
            writer.writerow(row)


def calculate_model_score(model_name: str):
    output_path = f"output/baseline/direct/{model_name}"
    bleurt_mt_score = defaultdict(dict)
    comet_mt_score = defaultdict(dict)

    for src in LANGS.keys():
        for trg in LANGS.keys():
            if src == trg:
                continue
            output_file = f"{output_path}/flores_test_{src}-{trg}.json"
            score = compute_score(output_file)
            bleurt_mt_score[src][trg] = score["BLEURT"]
            comet_mt_score[src][trg] = score["COMET"]

    print(f"Model: {model_name}")
    print("========== BLEURT SCORE =============")
    print(json.dumps(bleurt_mt_score, indent=2))
    bleurt_save_path = os.path.join(output_path, "bleurt_mt_score.csv")
    dump_csv_file(bleurt_mt_score, bleurt_save_path)

    print("\n========== COMET SCORE ==============")
    print(json.dumps(comet_mt_score, indent=2))
    comet_save_path = os.path.join(output_path, "comet_mt_score.csv")
    dump_csv_file(comet_mt_score, comet_save_path)


def calculate_model_score_accelerated(model_name: str):
    output_path = f"output/baseline/direct/{model_name}"
    bleurt_mt_score = defaultdict(dict)
    comet_mt_score = defaultdict(dict)

    cum_src_list = []
    cum_ref_list = []
    cum_pred_list = []
    lang_pairs = []
    for src in LANGS.keys():
        for trg in LANGS.keys():
            if src == trg:
                continue
            output_file = f"{output_path}/flores_test_{src}-{trg}.json"
            with open(output_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            src_list, ref_list, pred_list = [], [], []
            for d in data:
                src_list.append(d["source"])
                ref_list.append(d["reference"])
                pred = pred_post_process(d["predicted"])
                pred_list.append(pred)

            lang_pairs.append((src, trg))
            cum_src_list.append(src_list)
            cum_ref_list.append(ref_list)
            cum_pred_list.append(pred_list)

    bleurt_mt_scores = bleurt_scorer.batch_score(
        cum_ref_list, cum_pred_list, verbose=True
    )
    comet_mt_scores = comet_scorer.batch_score(cum_src_list, cum_pred_list)
    for bleurt_score, comet_score, (src_lang, trg_lang) in zip(
        bleurt_mt_scores, comet_mt_scores, lang_pairs
    ):
        bleurt_mt_score[src_lang][trg_lang] = bleurt_score
        comet_mt_score[src_lang][trg_lang] = comet_score

    print(f"Model: {model_name}")
    print("========== BLEURT SCORE =============")
    print(json.dumps(bleurt_mt_score, indent=2))
    bleurt_save_path = os.path.join(output_path, "bleurt_mt_score.csv")
    dump_csv_file(bleurt_mt_score, bleurt_save_path)

    print("\n========== COMET SCORE ==============")
    print(json.dumps(comet_mt_score, indent=2))
    comet_save_path = os.path.join(output_path, "comet_mt_score.csv")
    dump_csv_file(comet_mt_score, comet_save_path)


if __name__ == "__main__":
    bleurt_scorer = BleurtScorer(batch_size=16)
    comet_scorer = CometScorer(batch_size=32)
    for model in MODELS:
        calculate_model_score_accelerated(model)
