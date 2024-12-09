import csv
import os
from collections import defaultdict

import numpy as np

model_names = [
    "Llama-3.2-1B-Instruct",
    "Llama-3.2-3B-Instruct",
    "Llama-3.1-8B-Instruct",
]
OUTPUT_PATH = "output/baseline/direct"
METRICS = ("bleurt", "comet")

for model_name in model_names:
    for metric in METRICS:
        input_fpath = os.path.join(OUTPUT_PATH, model_name, f"{metric}_mt_score.csv")
        output_fpath = os.path.join(
            OUTPUT_PATH, model_name, f"{metric}_mt_score_modified.csv"
        )
        with open(input_fpath, "r", encoding="utf-8") as infile, open(
            output_fpath, "w+", encoding="utf-8"
        ) as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)
            for row in reader:
                new_row = []
                for element in row:
                    if element == "-1" or element == -1:
                        new_row.append("-")
                        continue
                    try:
                        ele = float(element) * 100
                        ele = np.round(ele, 2)
                        new_row.append(ele)
                    except ValueError:
                        new_row.append(element)
                writer.writerow(new_row)
