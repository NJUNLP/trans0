import sys

import bleurt
from bleurt import score as bleurt_score
from bleurt.lib.tokenizers import create_tokenizer

sys.argv = sys.argv[:1]
import json

import tensorflow.compat.v1 as tf
import torch
import torch.nn as nn
import transformers

references = ["a bird chirps by the window"]
candidates = ["a bird chirps by the window"]
checkpoint = "/mnt/bn/v2024/models/BLEURT-20/"
imported = tf.saved_model.load_v2(checkpoint)

state_dict = {}
for variable in imported.variables:
    n = variable.name
    if n.startswith('global'):
        continue
    data = variable.numpy()
    # if 'dense' in n:
    if 'kernel' in n:  # this is fix #1 - considering 'kernel' layers instead of 'dense'
        data = data.T
    n = n.split(':')[0]
    n = n.replace('/','.')
    n = n.replace('_','.')
    n = n.replace('kernel','weight')
    if 'LayerNorm' in n:
        n = n.replace('beta','bias')
        n = n.replace('gamma','weight')
    elif 'embeddings' in n:
        n = n.replace('word.embeddings','word_embeddings')
        n = n.replace('position.embeddings','position_embeddings')
        n = n.replace('token.type.embeddings','token_type_embeddings')
        n = n + '.weight'
    state_dict[n] = torch.from_numpy(data)

class BleurtModel(nn.Module):
    """
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.bert = transformers.RemBertModel(config)
        self.dense = nn.Linear(config.hidden_size,1)

    def forward(self, input_ids, input_mask, segment_ids):
        cls_state = self.bert(
            input_ids,
            input_mask,
            #   segment_ids)[0][:,0]#[1] doesnt work either
            segment_ids,
        ).pooler_output  # this is fix #2 - taking pooler output
        return self.dense(cls_state)

config = transformers.RemBertConfig()
bleurt_model = BleurtModel(config)
bleurt_model.load_state_dict(state_dict, strict=False)  # strict=False added otherwise crashes.
# Should be safe, according to this https://github.com/huggingface/transformers/issues/6882#issuecomment-884730078
for param in bleurt_model.parameters():
    param.requires_grad = False
bleurt_model.eval()

tf_scorer = bleurt_score.BleurtScorer(checkpoint)
scores = tf_scorer.score(references=references, candidates=candidates)
print(scores)


with open(f'{checkpoint}/bleurt_config.json','r') as f:
    bleurt_config = json.load(f)
max_seq_length = bleurt_config["max_seq_length"]
vocab_file = bleurt_config["vocab_file"]
do_lower_case = bleurt_config["do_lower_case"]
sp_model =  f'{checkpoint}/sent_piece'

tokenizer = create_tokenizer(
    vocab_file=None, do_lower_case=None, sp_model= sp_model)
input_ids, input_mask, segment_ids = bleurt.encoding.encode_batch(
          references, candidates, tokenizer, max_seq_length)

scores = bleurt_model(torch.from_numpy(input_ids),
             torch.from_numpy(input_mask),
             torch.from_numpy(segment_ids))
print(scores)
