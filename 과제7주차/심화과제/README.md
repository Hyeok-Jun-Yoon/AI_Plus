##7주차 심화 과제

* Argument

    --model_name_or_path "openai-community/openai-gpt" \
    --per_device_train_batch_size 8 \
    --do_train \
    --block_size 512 \
    --output_dir /tmp/advance-01 \
    --save_total_limit 1 \
    --logging_steps 10  \
    --evaluation_strategy "steps"  \
    --eval_steps 10  \
    --report_to "wandb"

1. train loss graph

https://wandb.ai/hyuckjun28-the-independent/Hanghae99/reports/train-loss-25-02-06-15-48-19---VmlldzoxMTIzMjgwMA
![image](https://github.com/user-attachments/assets/a2bf2145-d5a6-4b66-80db-f1a9b3172b20)

2. eval loss graph
 
https://wandb.ai/hyuckjun28-the-independent/Hanghae99/reports/eval-loss-25-02-06-15-48-45---VmlldzoxMTIzMjgwNQ
![image](https://github.com/user-attachments/assets/3f93d870-631e-497d-9b77-aaabf314e22e)
