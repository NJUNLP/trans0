from pathlib import Path
from typing import List

import numpy as np
from tqdm import trange

DEFAULT_BLEURT_CKPT = "models/huggingface/bleurt20"
DEFAULT_COMET_CKPT = "models/Unbabel/wmt22-cometkiwi-da/checkpoints/model.ckpt"


class AbstractScorer(object):
    def score(self, references: List[str], hypothesizes: List[str]):
        raise NotImplementedError


class BleurtScorer(AbstractScorer):
    def __init__(self, ckpt_path: Path = DEFAULT_BLEURT_CKPT, batch_size: int = 32):
        from bleurt_pytorch import BleurtForSequenceClassification, BleurtTokenizer

        self.bleurt_scorer = BleurtForSequenceClassification.from_pretrained(
            ckpt_path, device_map="cuda:0"
        )
        self.bleurt_tokenizer = BleurtTokenizer.from_pretrained(
            ckpt_path, device_map="cuda:0"
        )
        self.bleurt_scorer.eval()
        self.batch_size = batch_size

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


class CometScorer(AbstractScorer):
    def __init__(self, ckpt_path: Path = DEFAULT_COMET_CKPT, batch_size: int = 32):
        import comet

        self.comet_scorer = comet.load_from_checkpoint(ckpt_path, reload_hparams=True)
        self.batch_size = batch_size

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


if __name__ == "__main__":
    bleurt_scorer = BleurtScorer()
    comet_scorer = CometScorer()

    src_list = ["Hello, how are you?", "I'm fine, thank you."]
    ref_list = ["Bonjour, comment ça va?", "Je vais bien, merci."]
    hypo_list = ["Bonjour, comment ça?", "Je vais bien merci"]

    bleurt_score = bleurt_scorer.score(ref_list, hypo_list)
    comet_score = comet_scorer.score(src_list, hypo_list)

    print(f"Bleurt score: {bleurt_score}")
    print(f"Comet score: {comet_score}")
