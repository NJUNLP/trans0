from typing import List
from pathlib import Path
from tqdm import trange
import numpy as np
import torch

def calculate_cum_size(arrays):
    cum_size = []
    start_index = 0
    for array in arrays:
        end_index = start_index + len(array)
        cum_size.append([start_index, end_index])
        start_index = end_index
    return cum_size

class AbstractScorer(object):
    def score(self, references: List[str], hypothesizes: List[str]):
        raise NotImplementedError

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        """Compute score for multiple references and hypothesizes together"""
        raise NotImplementedError

class BleurtScorer(AbstractScorer):
    def __init__(self, ckpt_path: Path, batch_size: int = 32):
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

        self.bleurt_scorer = BleurtForSequenceClassification.from_pretrained(
            ckpt_path, device_map="cuda", trust_remote_code=True
        )
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
        self.bleurt_scorer.eval()
        self.batch_size = batch_size

    def load_cuda(self):
        self.bleurt_scorer = self.bleurt_scorer.to("cuda")

    def offload_cuda(self):
        self.bleurt_scorer = self.bleurt_scorer.to("cpu")
        torch.cuda.empty_cache()

    def _score(
        self, references: List[str], hypothesizes: List[str], verbose: bool = True
    ):
        scores = []
        for i in trange(
            0,
            len(references),
            self.batch_size,
            desc="BLEURT scoring",
            disable=not verbose,
        ):
            inputs = self.bleurt_tokenizer(
                references[i : i + self.batch_size],
                hypothesizes[i : i + self.batch_size],
                return_tensors="pt",
                padding="longest",
            ).to(self.bleurt_scorer.device)
            cur_scores = self.bleurt_scorer(**inputs).logits.flatten().tolist()
            if i == 0:
                scores = cur_scores
            else:
                scores += cur_scores
        return scores

    def score(self, references: List[str], hypothesizes: List[str]):
        """
        references: List of target sentences
        hypothesizes: List of MT results
        """
        scores = self._score(references, hypothesizes)
        score = np.array(scores).mean()
        return score

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        scores = []
        cum_size = calculate_cum_size(references)
        cum_references = [ref for refs in references for ref in refs]
        cum_hypothesizes = [hypo for hypos in hypothesizes for hypo in hypos]

        scores = self._score(cum_references, cum_hypothesizes)

        avg_scores = []
        for cum in cum_size:
            start, end = cum
            score = np.array(scores[start:end]).mean()
            avg_scores.append(score)
        return avg_scores

class CometScorer(AbstractScorer):

    def __init__(self, ckpt_path: Path, batch_size: int = 128):
        import comet

        self.comet_scorer = comet.load_from_checkpoint(ckpt_path, reload_hparams=True).eval()
        self.batch_size = batch_size

    def load_cuda(self):
        self.comet_scorer = self.comet_scorer.to("cuda")

    def offload_cuda(self):
        self.comet_scorer = self.comet_scorer.to("cpu")
        torch.cuda.empty_cache()

    def _score(self, references: List[str], hypothesizes: List[str]):
        data = [{"src": ref, "mt": hypo} for ref, hypo in zip(references, hypothesizes)]
        comet_output = self.comet_scorer.predict(
            data, batch_size=self.batch_size, gpus=1
        )
        scores = comet_output.scores
        return scores

    def score(self, references: List[str], hypothesizes: List[str]):
        """
        references: List of source sentences
        hypothesizes: List of MT results
        """
        scores = self._score(references, hypothesizes)
        score = np.mean(scores)
        return score

    def batch_score(self, references: List[List[str]], hypothesizes: List[List[str]]):
        cum_size = calculate_cum_size(references)
        cum_references = [ref for refs in references for ref in refs]
        cum_hypothesizes = [hypo for hypos in hypothesizes for hypo in hypos]

        scores = self._score(cum_references, cum_hypothesizes)

        avg_scores = []
        for cum in cum_size:
            start, end = cum
            score = np.array(scores[start:end]).mean()
            avg_scores.append(score)
        return avg_scores
