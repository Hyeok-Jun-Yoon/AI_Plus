import os
import sys
import math
import torch
import wandb
import logging
import datasets
import argparse
import evaluate
import transformers

from typing import Optional
from itertools import chain
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from transformers.trainer_utils import get_last_checkpoint

wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-finetuning-yoon-corpus'

@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)  # HuggingFace hub에서 pre-trained 모델로 사용할 모델의 이름
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})  # 우리 모델의 precision(data type이라고 이해하시면 됩니다)

    dataset_name: Optional[str] = field(default=None)  # Fine-tuning으로 사용할 huggingface hub에서의 dataset 이름
    dataset_config_name: Optional[str] = field(default=None)  # Fine-tuning으로 사용할 huggingface hub에서의 dataset configuration
    block_size: int = field(default=1024)  # Fine-tuning에 사용할 input text의 길이
    num_workers: Optional[int] = field(default=None)  # Data를 업로드하거나 전처리할 때 사용할 worker 숫자
    
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()

#training_args.report_to = 'wandb'
#training_args.report_to = "wandb"  # WandB로 로그 전송
training_args.evaluation_strategy = "steps"
training_args.eval_accumulation_steps = 1  # 평가 시 배치 크기 축적
training_args.eval_step = 10
training_args.logging_dir='/tmp/logs'

logger = logging.getLogger()

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()  # log level을 INFO로 변경 

log_level = training_args.get_process_log_level()

# 우리가 가지고 있는 logger와 HuggingFace의 logger의 log level 설정
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# 기타 HuggingFace logger option들을 설정
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

logger.info(f"Training/evaluation parameters {training_args}")

#corpus.json 파일 load
raw_datasets = load_dataset("json", data_files={"train": "/home/ubuntu/data/corpus.json", "validation": "/home/ubuntu/data/corpus.json"})


config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# pad_token 설정
tokenizer.pad_token = tokenizer.eos_token
tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

# Tokenizer가 이미 pad_token을 가지고 있는지 확인
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = '[PAD]'
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids('[PAD]')


embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

def tokenize_function(examples):
    input_ids = tokenizer(examples['Instruction'], padding="max_length", truncation=True, max_length=args.block_size)
    labels = tokenizer(examples['Response'], padding="max_length", truncation=True, max_length=args.block_size)
    input_ids["labels"] = labels["input_ids"]
    return input_ids

    
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=["Instruction", "Response"]
    )

max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)
  
# Train 및 validation data 준비
train_test_split = tokenized_datasets["train"].train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
validation_dataset = train_test_split["test"]

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)

checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)  # 만약 output_dir에 checkpoint가 남아있으면 이를 사용하고, 없으면 None이 return됩니다.
if training_args.resume_from_checkpoint is not None:  # output_dir이 아닌 다른 위치에서의 checkpoint를 resume_from_checkpoint로 지정할 수 있습니다.
    checkpoint = training_args.resume_from_checkpoint
else:  # 아니면 last_checkpoint로 checkpoint를 지정합니다.  
    checkpoint = last_checkpoint
    
train_result = trainer.train(resume_from_checkpoint=checkpoint)

trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

# 평가 지표를 계산합니다.
with torch.no_grad():
    eval_metrics = trainer.evaluate(eval_dataset=validation_dataset)

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
