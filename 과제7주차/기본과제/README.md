## 7주차 기본 과제

* Argument
    --model_name_or_path "openai-community/openai-gpt" \
    --per_device_train_batch_size 8 \
    --dataset_name "wikitext" \
    --dataset_config_name "wikitext-2-raw-v1" \
    --do_train \
    --output_dir /tmp/test-clm25 \
    --save_total_limit 1 \
    --logging_steps 10  \
	 --evaluation_strategy "steps"  \   
    --eval_steps 10


1. train loss graph
   
https://wandb.ai/hyuckjun28-the-independent/Hanghae99/reports/train-loss-25-02-06-09-05-16---VmlldzoxMTIyODk3MQ

![image](https://github.com/user-attachments/assets/34a2a1ab-e0f6-4410-a47c-8be9da1b8e46)


3. eval loss graph
   
https://wandb.ai/hyuckjun28-the-independent/Hanghae99/reports/eval-loss-25-02-06-09-05-30---VmlldzoxMTIyODk3Mw

![image](https://github.com/user-attachments/assets/73f17952-1ed7-4c7f-a207-f47ba6c159a4)
