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
from transformers.trainer_utils import get_last_checkpoint
# SFTTrainer 사용을 위한 import
from trl import SFTTrainer, SFTConfig

#LoRA 사용하기 위한 import
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)

wandb.init(project='Hanghae99')
wandb.run.name = 'LoRA-256-yoon'

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

training_args.evaluation_strategy = "steps"
training_args.eval_accumulation_steps = 1  # 평가 시 배치 크기 축적
training_args.eval_step = 10
training_args.logging_dir='/tmp/logs'

#LoRA 인자 추가 (lora_r을 8,128,256 일때로 변경해보면서 비교해보기)
lora_r: int = 256
lora_dropout: float = 0.1
lora_alpha: int = 32

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

# lucasmccabe-lmi/CodeAlpaca-20k 로드
raw_datasets = load_dataset(args.dataset_config_name)

config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# target_modules라는 변수에 LoRA를 적용할 module들을 저장
target_modules = set()

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        names = name.split('.')
        target_modules.add(names[0] if len(names) == 1 else names[-1])

if "lm_head" in target_modules:  # needed for 16-bit
    target_modules.remove("lm_head")

target_modules = list(target_modules)

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=lora_r,
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=target_modules
)
model = get_peft_model(model, peft_config)

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

max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

def formatting_prompts_func(examples):
    # 배치 내 각 항목에 대해 포맷팅
    formatted_examples = {
        'input': [],
        'output': []
    }
     # 예시가 배치로 들어오기 때문에, 각 리스트의 항목을 순차적으로 처리합니다.
    for instruction, input_text, output_text in zip(examples['instruction'], examples['input'], examples['output']):
        # 각 항목에 대해 포맷팅하여 딕셔너리의 리스트에 추가
        formatted_examples['input'].append(f"Q: {instruction}\n{input_text}")
        formatted_examples['output'].append(f"A: {output_text}")

    return formatted_examples

def tokenize(examples):
    input_texts = [str(text) for text in examples['input']]  # input을 문자열로 변환
    output_texts = [str(text) for text in examples['output']]  # output을 문자열로 변환

    encoding = tokenizer(input_texts, padding="max_length", truncation=True, max_length=args.block_size, return_tensors="pt")
    labels = tokenizer(output_texts, padding="max_length", truncation=True, max_length=args.block_size, return_tensors="pt")
    
    encoding['labels'] = labels['input_ids']
    return encoding

# 데이터셋 전처리
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        num_proc=args.num_workers
    )

collator = DataCollatorWithPadding(tokenizer)

# formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

# 5000줄만 선택 (전체 데이터로 진행해봤는데 4시간 넘게 걸려 일부만 사용)
subset_data = tokenized_datasets["train"].select(range(5000))

# 80%는 학습용, 20%는 검증용으로 나누기
train_size = int(0.8 * len(subset_data))  # 학습용 데이터의 크기
train_dataset = subset_data.select(range(train_size))  # 학습용 데이터
validation_dataset = subset_data.select(range(train_size, len(subset_data)))  # 검증용 데이터

'''
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    tokenizer=tokenizer,
    data_collator=default_data_collator
)
'''

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(output_dir="/tmp/clm-instruction-tuning", max_seq_length=128),
    eval_dataset=validation_dataset,
    formatting_func=formatting_prompts_func,
    data_collator=collator
)

trainer.train()
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

print('Max Alloc:', round(torch.cuda.max_memory_allocated(0)/1024**3, 1), 'GB')
