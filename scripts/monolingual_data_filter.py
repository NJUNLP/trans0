import json
import os

# other langs: 300k * 0.02 = 6k
# portuguese: 40k * 0.1 = 4k
LANG_PERCENT = {
    "zho_Hans": 0.02,
    "rus_Cyrl": 0.02,
    "por_Latn": 0.1,
    "ita_Latn": 0.02,
    "deu_Latn": 0.01,
}

for lang, percent in LANG_PERCENT.items():
    fpath = f"dataset/monolingual/{lang}/score.jsonl"
    with open(fpath, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    data = sorted(data, key=lambda x: x["score"])
    data = data[: int(len(data) * percent)]
    print(f"Processing {lang}..., {percent} percentage, get {len(data)} lines")

    trg_path = f"dataset/monolingual/{lang}/merged_filter.txt"
    with open(trg_path, "w+", encoding="utf-8") as f:
        for line in data:
            f.write(line["src"] + "\n")
