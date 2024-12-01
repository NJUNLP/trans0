import json
import os
from collections import defaultdict
from pathlib import Path

from modules.evaluation import BleurtScorer, CometScorer

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
    bleurt_save_path = os.path.join(output_path, "bleurt_mt_score.json")
    with open(bleurt_save_path, "w", encoding="utf-8") as f:
        json.dump(bleurt_mt_score, f, indent=2, ensure_ascii=False)

    print("\n========== COMET SCORE ==============")
    print(json.dumps(comet_mt_score, indent=2))
    comet_save_path = os.path.join(output_path, "comet_mt_score.json")
    with open(comet_save_path, "w", encoding="utf-8") as f:
        json.dump(comet_mt_score, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    bleurt_scorer = BleurtScorer()
    comet_scorer = CometScorer()
    for model in MODELS:
        calculate_model_score(model)
