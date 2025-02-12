## 기본과제: LoRA rank에 따른 학습 성능 비교해보기

모델 : facebook/opt-350m
데이터셋 : lucasmccabe-lmi/CodeAlpaca-20k (5000건만 진행/ train : 80%, eval : 20%)

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

![image](https://github.com/user-attachments/assets/733b0884-6044-4ce9-a544-5736a7140501)


차원을 늘려감에 따라 메모리 점유율은 향상되었고 학습 시간도 점점 증가됨을 확인 할 수 있었습니다.
train_loss는 소요시간이 너무 많이 걸려 전체 데이터로 하지 않고 일부만 진행해서 그런지 미세한 차이를 보였습니다. 

추가로 전체 데이터로 진행하고 lora_r값을 8로 진행했을때 train_loss 1.2091 로 내려간 것을 확인했습니다. (단지.. 학습 시간이 4시간이상 소요)
