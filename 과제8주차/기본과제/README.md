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
## 1. lora_r를 [8, 128, 256]로 변화시켜가며 학습 결과

![image](https://github.com/user-attachments/assets/b80fcf5a-2ed2-406d-821a-424565f24381)


## 2. 결과 그래프
**<lora_r 3개의 값 비교 그래프>**

![image](https://github.com/user-attachments/assets/162f6627-49ec-41d4-b899-9edccfa14f7e)

링크 : https://wandb.ai/hyuckjun28-the-independent/Hanghae99/reports/train-loss-25-02-12-11-01-02---VmlldzoxMTMxMzA1OA

**<lora_r 값이 8,128일때>**

![image](https://github.com/user-attachments/assets/7e356612-21fc-41ec-8103-dafd853920d4)

링크 : https://wandb.ai/hyuckjun28-the-independent/Hanghae99/reports/train-loss-25-02-12-11-04-57---VmlldzoxMTMxMzA5OA

**<lora_r 값이 256일때>**

![image](https://github.com/user-attachments/assets/30435175-fc4e-4614-94e7-66a1b7c6ea0e)

링크 : https://wandb.ai/hyuckjun28-the-independent/Hanghae99/reports/train-loss-25-02-12-11-05-48---VmlldzoxMTMxMzEwNA

## 결론
차원을 늘려감에 따라 메모리 점유율은 향상되었고 학습 시간도 점점 증가함을 확인 할 수 있었습니다.
학습시 소요시간이 너무 많이 걸려 전체 데이터로 하지 않고 일부만 진행해서 그런지 train_loss의 차이는 미세하였고,
r을 늘렸을 때 줄어드는 거 같더니 256차원으로 진행했을 때는 오히려 loss 값이 상승 됨을 보였습니다.
즉 모델의 크기와 복잡도에 따라 학습 시간이 길어지고 더 많은 메모리를 요구하는 것을 확인하였고 차원을 과도하게 증가하는 것은 오히려 학습 효율을 떨어트린다는 것을 알 수 있었습니다.

* 최적의 lora_r 값
**lora_r=8**에서 훈련 성능과 효율성을 고려했을 때 가장 적합하다고 볼 수 있을 거 같습니다.

추가로 전체 데이터로 진행하고 lora_r 값을 8로 진행했을 때 train_loss 1.2091로 내려간 것을 확인했습니다. (단지..학습 시간이 4시간 이상 진행됨.)

