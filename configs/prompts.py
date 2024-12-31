# -*- coding: utf-8 -*-
# for translation instruction, must contains the target language
TRANS_PROMPTS = [
    "Please translate the {src_lan} into {trg_lan}: {src_sent}",
    "Translate this from {src_lan} to {trg_lan}:\n{src_lan}: {src_sent}\n{trg_lan}: ",
    "{src_sent} in {src_lan} can be translated to {trg_lan} as: ",
    "{src_lan}: {src_sent} = {trg_lan}: ",
    "{src_lan}: {src_sent}\n{trg_lan}: ",
    "Explain the following {src_lan} sentence in {trg_lan}: {src_sent}",
]

def make_mt_instruction(instruction:str, tower_instruction:bool=False):
    """
    make instructions for instruct-version MT-agent tuning
    """
    if tower_instruction:
        message = [
            {"role": "user", "content": instruction},
        ]
    else:
        message=[
            {"role": "system","content": "You are a multilingual translator mastering several languages."},
            {"role": "user", "content": instruction},
        ]
    return message

# similar sentence prompt
SIM_SENT_PROMPT = "please generate a similar sentence:"

# to mark the generation results (Llama-instruct use </LABEL>)
LABEL_MARK = "<LABEL>"

# translate with examplars (translation history)
TRANS_CONTEXT_PROMPT =  [
    "{src_lan}: {word_pair_src} = {trg_lan}: {word_pair_trg}\n",
    "For example, {src_lan}: {word_pair_src} = {trg_lan}: {word_pair_trg}\n",
]


# evaluation prompts
EVAL_TRANS_PROMPT = """You are a multilingual language expert providing clarity evaluation of 0-6 points for translation by following criteria:
0: the translation is nonsensical, failing to convey any coherent meaning.
2: the translation partially preserves the meaning of source text, albeit with substantial inaccuracies or omissions.
4: the translation largely maintains the source text's meaning, with only minor issues such as slight grammar errors.
6: A perfect translation accurately conveying the full meaning of the source text without any errors.
The format of all inputs is in JSON.
# The translation pair:
{"{src_lan}": {src_sent}, "{trg_lan}": {trg_sent}}
# Score:
"""

RM_PROMPT = "Translate the following sentence from <src_lan> to <trg_lan>:\n<src_sent> <end>"
