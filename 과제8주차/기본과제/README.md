## 기본과제: LoRA rank에 따른 학습 성능 비교해보기

1. lora_r를 [8, 128, 256]로 변화시켜가며 학습
python basic_8.py \           
    --model_name_or_path "facebook/opt-350m" \
    --per_device_train_batch_size 8 \
    --do_train \
    --block_size 512 \
    --torch_dtype "bfloat16"  \
    --output_dir /tmp/clm-instruction-tuning/02 \
    --dataset_config_name "lucasmccabe-lmi/CodeAlpaca-20k" \
    --save_total_limit 1 \
    --logging_steps 10  \
    --evaluation_strategy "steps"  \
    --eval_steps 10  \
    --report_to "wandb"
