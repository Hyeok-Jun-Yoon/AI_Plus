## Q1) 어떤 task를 선택하셨나요?
> MNLI 선택했습니다.


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 기존의 구현된 구조를 활용하여 입력은 두 문장을 하나의 문장으로 합쳐 중간에 [SEP] 를 추가해서 input을 구성했습니다.
  출력 label은 3개로  0: 모순 / 1 : 연결  / 2 : 무관 형태로 나오게 됩니다.


## Q3) 어떤 pre-trained 모델을 활용하셨나요?
> DistilBERT를 활요하여 MNLI 구현했습니다.


## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> epoch 50번을 학습 시킨 결과 loss 가 169.6 ~ 151.9 로 떨어졌고 아래 그림같이 구간안에서 수렴하는 듯 보였습니다.
> 정확도를 구했을때는 Train acc: 0.554 | Test acc: 0.491 로 나왔습니다.
> pre-train 하지 않은 transformer 로 학습했을때는 223.5 ~ 171.9 점대에서 머물렸습니다.
> 정확도는 =========> Train acc: 0.369 | Test acc: 0.327 로 낮은 비율을 보였습니다.

- 이미지 첨부시 : ![train_loss비교](https://github.com/Hyeok-Jun-Yoon/AI_Plus/blob/main/%EC%8B%AC%ED%99%94%EA%B3%BC%EC%A0%9C3%EC%A3%BC%EC%B0%A8/train_loss%EB%B9%84%EA%B5%90.png)

### 위의 사항들을 구현하고 나온 결과들을 정리한 보고서를 README.md 형태로 업로드
### 코드 및 실행 결과는 jupyter notebook 형태로 같이 public github repository에 업로드하여 공유해주시면 됩니다(반드시 출력 결과가 남아있어야 합니다!!) 
