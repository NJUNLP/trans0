from modules.metrics import *

DEFAULT_BLEURT_CKPT = "/mnt/bn/v2024/models/huggingface/bleurt20"
DEFAULT_COMET_CKPT = "/mnt/bn/v2024/models/Unbabel/wmt22-cometkiwi-da/checkpoints/model.ckpt"

if __name__ == "__main__":
    bleurt_scorer = BleurtScorer(DEFAULT_BLEURT_CKPT)
    comet_scorer = CometScorer(DEFAULT_COMET_CKPT)

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
