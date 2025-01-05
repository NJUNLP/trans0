import json
import os

import matplotlib.pyplot as plt
import numpy as np

LANG = ["deu_Latn", "ita_Latn", "por_Latn", "rus_Cyrl", "zho_Hans"]


def analyze(data: list[dict], lang: str):
    score_list = [round(d["score"] * 100) for d in data]
    bins = np.linspace(0, 100, 51)  # 生成 0 到 100 的 20 个区间
    counts, edges = np.histogram(score_list, bins=bins)

    # 绘制柱状图
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(
        edges[:-1],
        counts,
        width=np.diff(edges),
        align="edge",
        color="lightblue",
        edgecolor="black",
    )
    ax1.set_xlabel("Score Range")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"{lang}-Eng_Latn Score Distribution")

    # 绘制累计分布曲线
    ax2 = ax1.twinx()  # 创建一个共享 x 轴的第二个 y 轴
    cumulative_counts = np.cumsum(counts)
    cumulative_counts = np.round(cumulative_counts / len(score_list), 2) * 100
    ax2.plot(
        edges[1:],
        cumulative_counts,
        color="red",
        marker="o",
        label="Cumulative Distribution",
        lw=2,
        markersize=3,
    )
    ax2.set_ylabel("Cumulative Frequency")
    ax2.set_ylim(0, max(cumulative_counts) * 1.1)

    # 在右侧显示区间的累计分布
    for i in range(len(counts)):
        ax2.text(
            edges[i] + 2,
            cumulative_counts[i],
            f"{cumulative_counts[i]}".split(".")[0],
            ha="center",
            va="bottom",
            fontsize=6,
        )

    # 显示图表
    fig.tight_layout()
    return plt


def main():
    for lang in LANG:
        dir_path = f"dataset/monolingual/{lang}"
        fpath = os.path.join(dir_path, "score.jsonl")
        with open(fpath, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f.readlines()]
            plt = analyze(data, lang)
            img_path = os.path.join(dir_path, "score.png")
            plt.savefig(img_path)


if __name__ == "__main__":
    main()
