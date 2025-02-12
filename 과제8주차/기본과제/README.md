## 기본과제: LoRA rank에 따른 학습 성능 비교해보기

모델 : facebook/opt-350m
데이터셋 : lucasmccabe-lmi/CodeAlpaca-20k

실행 Argument
    --model_name_or_path "facebook/opt-350m" \
    --per_device_train_batch_size 8 \
    --do_train \
    --block_size 512 \
    --torch_dtype "bfloat16"  \
    --output_dir /tmp/clm-instruction-tuning \
    --dataset_config_name "lucasmccabe-lmi/CodeAlpaca-20k" \
    --save_total_limit 1 \
    --logging_steps 10  \
    --evaluation_strategy "steps"  \
    --eval_steps 10  \
    --report_to "wandb"
1. lora_r를 [8, 128, 256]로 변화시켜가며 학습

lora_r 	8	128	256
train/loss	1.2592	1.259	
eval/loss	1.2012	1.2015	
train_runtim	1:07:40	1:13:11	
Memory	9.7 GB	10.5GB	
![image](https://github.com/user-attachments/assets/733b0884-6044-4ce9-a544-5736a7140501)
