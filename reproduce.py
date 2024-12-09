from transformers import AutoTokenizer
from trl.trainer.utils import DataCollatorForChatML

tokenizer = AutoTokenizer.from_pretrained("/Users/nil/Downloads/Llama-3.2-1B-Instruct")
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.pad_token = tokenizer.eos_token

data_collator = DataCollatorForChatML(tokenizer)
examples = [
    {
        "messages": [
            {"role": "system", "content": "You are a professional translator."},
            {"role": "user", "content": "Hello!"},
            {"role": "assistant", "content": "Hi there! How can I help you today?"},
        ],
    },
]
batch = data_collator(examples)

print(tokenizer.decode(batch["input_ids"][0]))

label = batch["labels"][0]
label[label == -100] = 0
print(tokenizer.decode(label))
