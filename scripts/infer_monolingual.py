import json
import multiprocessing
import os
import sys

import torch.distributed as dist
import transformers
from lingua import LanguageDetectorBuilder
from tqdm import tqdm
from vllm import SamplingParams

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from configs.configs import DefaultTrainingArguments
from configs.lang_codes import LangCodes
from configs.prompts import LABEL_MARK, TRANS_PROMPTS, make_mt_instruction
from modules.inference import lang_codes, prepare_vllm_inference
from modules.metrics import CometScorer
from utils.common_utils import free_gpu, get_path

# TEST_DATA_SIZE = 2000
TEST_DATA_SIZE = 300000
NON_ENGLISH_PUNISH = 0.5

language_detector = LanguageDetectorBuilder.from_all_languages().build()
supported_langs = LangCodes()


def load_data(lang: str):
    data_path = f"dataset/monolingual/{lang}/merged.txt"
    trans_prompt = TRANS_PROMPTS[0]
    raw_inputs = []
    input_lists = []

    idx = 0
    with open(data_path, "r", encoding="utf-8") as f:
        line = f.readline()
        while line:
            line = line.strip()
            raw_inputs.append(line)
            input_line = trans_prompt.format(
                src_lan=lang_codes.get_lang(lang), trg_lan="English", src_sent=line
            )
            input_lists.append(input_line)

            idx += 1
            if idx > TEST_DATA_SIZE:
                break
            line = f.readline()
    return raw_inputs, input_lists


def check_lang_english(text: str) -> bool:
    detect_code = language_detector.detect_language_of(text)
    if detect_code is None:
        return False
    else:
        detect_language = supported_langs.get_lang(
            detect_code.iso_code_639_1.name.lower()
        )
        return detect_language == "English"


def process_data(src_hypo_score):
    src, hypo, score = src_hypo_score
    is_english = check_lang_english(hypo)
    if not is_english:
        score *= NON_ENGLISH_PUNISH
    data = {
        "src": src,
        "hypo": hypo,
        "score": score,
    }
    return json.dumps(data, ensure_ascii=False)


def generate(args, lang: str, llm, model_dir=None):
    if not os.path.exists(os.path.join(get_path(args, args.cache_dir))):
        os.makedirs(os.path.join(get_path(args, args.cache_dir)))
    raw_inputs, input_lists = load_data(lang)

    # prepare VLLM inference
    sampling_params = SamplingParams(
        n=1, temperature=0.0, max_tokens=args.max_new_tokens
    )
    tokenizer = llm.get_tokenizer()
    if tokenizer.chat_template is not None:
        input_lists = [
            tokenizer.apply_chat_template(
                make_mt_instruction(input_l, llm_path=tokenizer.name_or_path), tokenize=False, add_generation_prompt=True
            )
            for input_l in input_lists
        ]
    elif "ALMA" not in args.output_dir:
        input_lists = [input_l + LABEL_MARK for input_l in input_lists]
    generation_out = llm.generate(input_lists, sampling_params=sampling_params)
    cached_out_lists = []  # cached for metric evaluation
    cached_output_dir = os.path.join(get_path(args, args.cache_dir), "filter_output")
    os.makedirs(cached_output_dir, exist_ok=True)
    cached_out_path = os.path.join(cached_output_dir, f"{lang}-merged-en.txt")
    with open(cached_out_path, "w", encoding="utf-8") as out_file:
        for item in generation_out:
            for item_out in item.outputs:
                l = item_out.text
                out_file.write(l.strip() + "\n")
                cached_out_lists.append(l.strip())
    print("finished")
    print(">> test snipet>>", input_lists[0], ">> output>>", cached_out_lists[0])
    return raw_inputs, cached_out_lists


def validate(
    args,
    comet_scorer,
    lang: str,
    raw_inputs: list[str],
    cached_out_lists: list[str],
    model_dir=None,
):
    """
    validate by dev_data_path file, log the validation by global_step when it's not None
    :param valid_type: the type of the validation, ["en-x", "x-x", "x-en", "all"]
    :param dev_data_path: a parallel data file. If None, will extract flores.py for multi-lingual parallel test
    :param valid_type: the type of the validation, ["input_lang_code", "input", "output_lang_code", "output"]
    :param model_dir: the model to validate, if None, validate the model in args.output_dir
    """
    # comet_scorer = CometScorer(ckpt_path=get_path(args, args.comet_ckpt))
    comet_score = comet_scorer._score(raw_inputs, cached_out_lists)

    dump_file_path = f"dataset/monolingual/{lang}/score.jsonl"
    src_hypo_score_list = list(zip(raw_inputs, cached_out_lists, comet_score))

    with multiprocessing.Pool(processes=10) as pool:
        results = list(
            tqdm(
                pool.imap(process_data, src_hypo_score_list),
                total=len(src_hypo_score_list),
                desc=f"LANG: {lang}",
            )
        )

    with open(dump_file_path, "w+", encoding="utf-8") as f:
        for line in results:
            f.write(line + "\n")


def main():
    parser = transformers.HfArgumentParser(DefaultTrainingArguments)
    args = parser.parse_args()

    os.environ["HF_HOME"] = os.path.join(args.nas_base_path, "cache")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(args.nas_base_path, "cache")
    os.environ["NCCL_DEBUG"] = "INFO"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    args = parser.parse_args_into_dataclasses()[0]
    llm = prepare_vllm_inference(args, model_dir=None, override_cache=True)
    data = dict()
    for lang in args.self_play_languages:
        raw_inputs, output_lists = generate(args, lang, llm)
        data[lang] = (raw_inputs, output_lists)
    del llm
    dist.destroy_process_group()
    free_gpu()

    comet_scorer = CometScorer(ckpt_path=get_path(args, args.comet_ckpt))
    for lang in args.self_play_languages:
        raw_inputs, output_lists = data[lang]
        validate(args, comet_scorer, lang, raw_inputs, output_lists)
    del comet_scorer
    free_gpu()


if __name__ == "__main__":
    main()
