from typing import List, Optional

from dataclasses import dataclass, field
from transformers import TrainingArguments
from peft import LoraConfig

nas_path = "/mnt/bn/v2024/"

peft_config = LoraConfig(
    r=64, lora_alpha=128,    
    lora_dropout=0.05, bias="none", 
    task_type="CAUSAL_LM",
    target_modules = ["q_proj","v_proj", "o_proj"],
)

sp_peft_config = LoraConfig(
    r=64, lora_alpha=128,
    lora_dropout=0.1, bias="none",
    task_type="CAUSAL_LM",
    target_modules = "all-linear" #["q_proj","v_proj", "o_proj"],
)

@dataclass
class DefaultTrainingArguments(TrainingArguments): 
    # tokenizer and data params
    train_data_path: str = field(
        default="dataset/nist_zh-en/json/",
        metadata={"help": "the path to training data"}
    )
    dev_data_path: Optional[str] = field(
        default=None, metadata={"help": "dev data for mode RL"}
    )
    test_data_path:  Optional[str] = field(
        default=None, metadata={"help": "test inputs for mode test"}
    )
    
    flores_script: Optional[str] = field(
        default=None, metadata={"help": "multilingual /monolingual inputs"}
    )

    nas_base_path: str = field(
        default=nas_path,  
        metadata={"help": "the base nas path for default paths"}
    )

    max_length: int = field(
        default=2048,
        metadata={"help": "the max sentence sequence length."}
    )   
    padding_side: str = field(
        default="left",
        metadata={"help": "the side for tokenizer to add padding tokens."}
    )
    truncation_side: str = field(
        default="left",
        metadata={"help": "the side for tokenizer truncate sequence."}
    )
    add_sep_token: bool =field(
        default=False,
        metadata={"help": "whether add a <sep> token between query and response."}
    )
    resize_vocab:  bool =field(
        default=False,
        metadata={"help": "whether resize the vocabulary to add special pad token for llama."}
    )
    remove_unused_columns: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )

    # model params
    pooling_type: str = field(
        default="average",
        metadata={"help": "the pooling for MC value estimation, selected from [average, max, last]."}
    )
    llm_path: str = field(
        default="models/huggingface/Llama-2-7b-hf", 
        metadata={"help": "the LLM path."}
    )

    # training hyperparams
    debug_mode: bool = field(
        default=False,
        metadata={"help": "whether use the debug mode."}
    )
    cache_dir: Optional[str] = field(default="cache" , metadata={"help": "path to cache"})
    optim: str = field(default="adamw_torch", metadata={"help": "the paramter to use"})
    use_lora: bool=field(default=True)
    learning_rate: float = field(default=1e-4, metadata={"help":"default lr is apt for lora sft"})
    lr_scheduler_type: str= field(default="cosine")

    rl_loss_type:str=field(default="sppo_hard")
    rl_learning_rate: float=field(default=1e-5, metadata={"help":"default lr is for lora tuning"})
    rl_lr_scheduler_type: str=field(default="constant")

    save_steps: float = field(default=10)
    save_total_limit: float=field(default=2)
    num_train_epochs: int=field(default=1)

    
    clip_range: float = field(default=0.2, metadata={"help": "the range to clip the importance reweighting ratio for policy optimization."})
    length_penalty: float = field(default=1., metadata={"help": "the penalty for seq length."})
    lm_sft_coeff: float = field(default=0., metadata={"help": "the coefficient for SFT data language modeling loss."})            
    lm_kl_coeff: float = field(default=0., metadata={"help": "the coefficient of kl regularizer."})

    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "training batch on each device"}
    )   
    gradient_accumulation_steps: int = field(
        default=64,
        metadata={"help": "accumualate to simulate large batch"}
    )

    valid_data_size: int = field(
        default=0,
        metadata={"help": "the data size for validation data"}
    )

    resume_from_checkpoint: Optional[str] = field(
        default=None, 
        metadata={"help":  "either training checkpoint or final adapter"}
    )
    # generation parameters:
    max_new_tokens: int = field(
        default=200,
        metadata={"help": "the max length for sentence-level translation."}
    )   

    # metric
    bleurt_ckpt: str = field(
        default="models/huggingface/bleurt20/",
        metadata={"help": "default bleurt=BLEURT-20 (torch version) as suggested"}
    )
    comet_ckpt: str = field(
        default="models/Unbabel/wmt22-cometkiwi-da/checkpoints/model.ckpt",
        metadata={"help":"default reference-free comet as suggested"} 
    )
    
    mcts_sample_size: int = field(
        default=4,
        metadata={"help":"expansion size for MCTS"}
    )