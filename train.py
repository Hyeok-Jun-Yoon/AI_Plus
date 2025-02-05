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

# WandB (Weights & Biases) 초기화. 실험 결과를 추적하고 시각화하는 데 사용됩니다.
wandb.init(project='Hanghae99')
wandb.run.name = 'gpt-finetuning'  # WandB run 이름 설정

# 데이터 클래스를 사용하여 명령행 인자를 정의합니다.
@dataclass
class Arguments:
    model_name_or_path: Optional[str] = field(default=None)  # HuggingFace Hub에서 사전 학습된 모델을 지정합니다.
    torch_dtype: Optional[str] = field(default=None, metadata={'choices': ['auto', 'bfloat16', 'float16', 'float32']})  # 모델의 데이터 타입(precision)을 설정합니다.
    dataset_name: Optional[str] = field(default=None)  # HuggingFace Hub의 데이터셋 이름입니다.
    dataset_config_name: Optional[str] = field(default=None)  # HuggingFace Hub의 데이터셋 설정 이름입니다.
    block_size: int = field(default=1024)  # 훈련에 사용할 입력 텍스트의 길이입니다.
    num_workers: Optional[int] = field(default=None)  # 데이터 로드 및 전처리에 사용할 worker 수입니다.

# HfArgumentParser를 사용하여 명령행 인자를 파싱합니다.
parser = HfArgumentParser((Arguments, TrainingArguments))
args, training_args = parser.parse_args_into_dataclasses()
training_args.per_device_train_batch_size = 2
training_args.per_device_eval_batch_size = 2
training_args.gradient_accumulation_steps = 8
training_args.eval_steps = 1000

# 로거를 초기화합니다.
logger = logging.getLogger()

# 로깅 설정을 구성합니다.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# training_args에 따라 로그 레벨을 설정합니다.
if training_args.should_log:
    transformers.utils.logging.set_verbosity_info()

log_level = training_args.get_process_log_level()

# 로거의 로그 레벨을 설정합니다.
logger.setLevel(log_level)
datasets.utils.logging.set_verbosity(log_level)
transformers.utils.logging.set_verbosity(log_level)

# Hugging Face 로거 옵션을 설정합니다.
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()

# 훈련/평가 파라미터를 로깅합니다.
logger.info(f"Training/evaluation parameters {training_args}")

# Hugging Face Hub에서 데이터셋을 로드합니다.
raw_datasets = load_dataset(
    args.dataset_name,
    args.dataset_config_name
)

# 사전 학습된 모델의 config, tokenizer, model을 로드합니다.
config = AutoConfig.from_pretrained(args.model_name_or_path)
tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    args.model_name_or_path,
    config=config,
    torch_dtype=args.torch_dtype
)

# 패딩 토큰을 EOS 토큰으로 설정합니다.
tokenizer.pad_token_id = tokenizer.eos_token_id

# 채팅 템플릿을 설정합니다.
tokenizer.chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}"

# 토큰 임베딩 사이즈를 조정합니다.
embedding_size = model.get_input_embeddings().weight.shape[0]
if len(tokenizer) > embedding_size:
    model.resize_token_embeddings(len(tokenizer))

# 데이터셋의 컬럼 이름을 가져옵니다.
column_names = list(raw_datasets["train"].features)
text_column_name = "text" if "text" in column_names else column_names[0]

# 텍스트를 토큰화하는 함수입니다.
def tokenize_function(examples):
    output = tokenizer(examples[text_column_name])
    return output

# 데이터셋을 토큰화합니다.
with training_args.main_process_first(desc="dataset map tokenization"):
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.num_workers,
        remove_columns=column_names
    )

# 모델의 최대 위치 임베딩 길이를 가져옵니다.
max_pos_embeddings = config.max_position_embeddings if hasattr(config, "max_position_embeddings") else 1024
block_size = args.block_size if tokenizer.model_max_length is None else min(args.block_size, tokenizer.model_max_length)

# 텍스트를 그룹화하는 함수입니다.
def group_texts(examples):
    # 텍스트를 연결합니다.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}

    # 전체 길이를 계산합니다.
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // block_size) * block_size

    # 텍스트를 block_size 단위로 나눕니다.
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # labels를 input_ids와 동일하게 설정합니다. (다음 토큰 예측)
    result["labels"] = result["input_ids"].copy()
    return result

# 텍스트를 그룹화합니다.
with training_args.main_process_first(desc="grouping texts together"):
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.num_workers
    )

# 훈련 데이터셋과 검증 데이터셋을 설정합니다.
#train_dataset = lm_datasets["train"]
train_dataset = lm_datasets["train"].train_test_split(test_size=0.5)["train"]
eval_dataset = lm_datasets["validation"]

# Trainer를 초기화합니다.
def compute_metrics(p):
    predictions, labels = p
    predictions = predictions.argmax(axis=-1)
    # 예시: 정확도 계산
    accuracy = (predictions == labels).mean()
    return {"accuracy": accuracy}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 검증 데이터셋 추가
    tokenizer=tokenizer,
    data_collator=default_data_collator,  # 데이터 collator는 필요에 따라 custom하게 정의할 수 있습니다. 여기서는 default collator를 사용합니다.
    compute_metrics=compute_metrics  # compute_metrics 함수 추가
)

# 체크포인트를 설정합니다.
checkpoint = None
last_checkpoint = get_last_checkpoint(training_args.output_dir)
if training_args.resume_from_checkpoint is not None:
    checkpoint = training_args.resume_from_checkpoint
else:
    checkpoint = last_checkpoint

# 모델 훈련
train_result = trainer.train(resume_from_checkpoint=checkpoint)

# 훈련된 모델을 저장합니다.
trainer.save_model()

# 훈련 결과를 로깅하고 저장합니다.
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

torch.cuda.empty_cache()

# 평가 지표를 계산합니다.
with torch.no_grad():
    eval_metrics = trainer.evaluate(eval_dataset=eval_dataset)

# 평가 지표를 Hugging Face 로거에 로깅
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

# WandB로 loss 로그 기록
wandb.log({"train_loss": metrics["train_loss"], "eval_loss": eval_metrics["eval_loss"]})