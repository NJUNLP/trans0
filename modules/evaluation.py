from pathlib import Path
from typing import List

import numpy as np
import torch
from tqdm import trange

DEFAULT_BLEURT_CKPT = "models/huggingface/bleurt20"
DEFAULT_COMET_CKPT = "models/Unbabel/wmt22-cometkiwi-da/checkpoints/model.ckpt"


class AbstractScorer(object):
    def score(self, references: List[str], hypothesizes: List[str]):
        raise NotImplementedError

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        """Compute score for multiple references and hypothesizes together"""
        raise NotImplementedError


def calculate_cum_size(arrays):
    cum_size = []
    start_index = 0
    for array in arrays:
        end_index = start_index + len(array)
        cum_size.append([start_index, end_index])
        start_index = end_index
    return cum_size


class BleurtScorer(AbstractScorer):
    def __init__(self, ckpt_path: Path = DEFAULT_BLEURT_CKPT, batch_size: int = 32):
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

        self.bleurt_scorer = BleurtForSequenceClassification.from_pretrained(
            ckpt_path, device_map="cuda:0"
        )
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(ckpt_path)
        self.bleurt_scorer.eval()
        self.batch_size = batch_size

    def load_cuda(self):
        self.bleurt_scorer = self.bleurt_scorer.to("cuda:0")

    def offload_cuda(self):
        self.bleurt_scorer = self.bleurt_scorer.to("cpu")
        torch.cuda.empty_cache()

    def score(self, references: List[str], hypothesizes: List[str]):
        """
        references: List of target sentences
        hypothesizes: List of MT results
        """
        scores = []
        for i in trange(0, len(references), self.batch_size, desc="BLEURT scoring"):
            inputs = self.bleurt_tokenizer(
                references[i : i + self.batch_size],
                hypothesizes[i : i + self.batch_size],
                return_tensors="pt",
                padding="longest",
            ).to("cuda:0")
            cur_scores = self.bleurt_scorer(**inputs).logits.flatten().tolist()
            if i == 0:
                scores = cur_scores
            else:
                scores += cur_scores

        score = np.array(scores).mean()
        return score

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        scores = []
        cum_size = calculate_cum_size(references)
        cum_references = [ref for refs in references for ref in refs]
        cum_hypothesizes = [hypo for hypos in hypothesizes for hypo in hypos]

        for i in trange(
            0, len(cum_hypothesizes), self.batch_size, desc="BLEURT scoring"
        ):
            inputs = self.bleurt_tokenizer(
                cum_references[i : i + self.batch_size],
                cum_hypothesizes[i : i + self.batch_size],
                return_tensors="pt",
                padding="longest",
            ).to("cuda:0")
            cur_scores = self.bleurt_scorer(**inputs).logits.flatten().tolist()
            if i == 0:
                scores = cur_scores
            else:
                scores += cur_scores
        avg_scores = []
        for cum in cum_size:
            start, end = cum
            score = np.array(scores[start:end]).mean()
            avg_scores.append(score)
        return avg_scores


class CometScorer(AbstractScorer):

    def __init__(self, ckpt_path: Path = DEFAULT_COMET_CKPT, batch_size: int = 128):
        import comet

        self.comet_scorer = comet.load_from_checkpoint(ckpt_path, reload_hparams=True)
        self.batch_size = batch_size

    def load_cuda(self):
        self.comet_scorer = self.comet_scorer.to("cuda:0")

    def offload_cuda(self):
        self.comet_scorer = self.comet_scorer.to("cpu")
        torch.cuda.empty_cache()

    def score(self, references: List[str], hypothesizes: List[str]):
        """
        references: List of source sentences
        hypothesizes: List of MT results
        """
        data = [{"src": ref, "mt": hypo} for ref, hypo in zip(references, hypothesizes)]
        comet_output = self.comet_scorer.predict(
            data, batch_size=self.batch_size, gpus=1
        )
        score = comet_output.system_score
        return score

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        cum_size = calculate_cum_size(references)
        cum_references = [ref for refs in references for ref in refs]
        cum_hypothesizes = [hypo for hypos in hypothesizes for hypo in hypos]
        data = [
            {"src": ref, "mt": hypo}
            for ref, hypo in zip(cum_references, cum_hypothesizes)
        ]
        comet_output = self.comet_scorer.predict(
            data, batch_size=self.batch_size, gpus=1
        )
        scores = comet_output.scores
        avg_scores = []
        for cum in cum_size:
            start, end = cum
            score = np.array(scores[start:end]).mean()
            avg_scores.append(score)
        return avg_scores


if __name__ == "__main__":
    bleurt_scorer = BleurtScorer()
    comet_scorer = CometScorer()

    src_list = [
        ["Hello, how are you?", "I like fried chickens."],
        ["I'm fine, thank you."],
    ]
    ref_list = [
        ["Bonjour, comment ça va?", "J'aime les poulets frits."],
        ["Je vais bien, merci."],
    ]
    hypo_list = [
        ["Bonjour, comment ça?", "Je n'aime pas les poulets frits."],
        ["Je vais bien merci"],
    ]

    bleurt_score = bleurt_scorer.batch_score(ref_list, hypo_list)
    comet_score = comet_scorer.batch_score(src_list, hypo_list)
    print(f"Bleurt score: {bleurt_score}")
    print(f"Comet score: {comet_score}")

    src_list = sum(src_list, [])
    ref_list = sum(ref_list, [])
    hypo_list = sum(hypo_list, [])
    bleurt_score = bleurt_scorer.score(ref_list, hypo_list)
    comet_score = comet_scorer.score(src_list, hypo_list)
    print(f"Bleurt score: {bleurt_score}")
    print(f"Comet score: {comet_score}")
