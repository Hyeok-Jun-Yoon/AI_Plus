## Q1) 어떤 task를 선택하셨나요?
> MNLI 선택했습니다.


## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 기존의 구현된 구조를 활용하여 입력은 두 문장을 하나의 문장으로 합쳐 중간에 [SEP] 를 추가해서 input을 구성했습니다.
  출력 label은 3개로  0: 모순 / 1 : 연결  / 2 : 무관 형태로 나오게 됩니다.


## Q3) 어떤 pre-trained 모델을 활용하셨나요?
> model_name = 'distilbert-base-uncased'
  
  encoder_model = DistilBertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
  
  transfomer 라이브러리 중 DistilBertForSequenceClassification을 활용하였습니다.

## Q4) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요?

1. Train Loss 비교

> pre-trained 모델을 fine-tuning 했을 때 loss curve는 epoch을 수행하면서 점진적으로 감소하는 폭을 그렸습니다.

> pre-trained 한 것으로 학습한 것과 pre-train 하지 않은것으로 학습한 train loss 의 차이는 3가지 정도 확인 해볼 수 있었습니다.


- ![train_loss비교](https://github.com/Hyeok-Jun-Yoon/AI_Plus/blob/main/%EC%8B%AC%ED%99%94%EA%B3%BC%EC%A0%9C3%EC%A3%BC%EC%B0%A8/pre_train_vs_non_pre_train_loss%EB%B9%84%EA%B5%90.png)


- 초기 Loss 값 비교:

  - Pre-trained 모델의 초기 Loss는 약 164 근처에서 시작하는 반면, Non Pre-trained 모델의 초기 Loss는 약 251 정도에서 시작합니다.
  이는 Pre-trained 모델이 초기 가중치를 더 효율적으로 설정했음을 보여줍니다.

- Loss 감소 패턴:

  - Pre-trained 모델은 epoch을 수행하면서 점차 일정한 간격으로 감소하며 빠르게 수렴한 것을 볼 수 있었습니다.
  Non Pre-trained 모델은 초기에 손실이 급격히 감소하지만 이후 학습 속도가 매우 느려지며, 170 부근에서 수렴합니다.

- 최종 손실 값 비교:

  - Pre-trained 모델은 약 150 부근에서 최종 손실 값을 가지며 Non Pre-trained 모델보다 낮습니다.
  Non Pre-trained 모델은 약 170 근처에서 학습이 수렴합니다.
  이는 Pre-trained 모델이 Non Pre-trained 모델보다 최적화에 더 효율적이라는 것을 볼 수 있습니다.

***
2. 정확도 비교
  - pre-trained 모델로 학습 시 정확도 Train acc: 0.554 | Test acc: 0.491 로 나왔습니다.
  - Non Pre-trained 모델은 Train acc: 0.369 | Test acc: 0.327 로 정확도 측면에서도 낮은 것을 알 수 있었습니다.

3. 결론
  - Pre-trained 모델은 초기 학습과 최종 성능 모두에서 Non Pre-trained 모델보다 훨씬 효율적이며, 사전 학습된 가중치가 더 낮은 손실 값을 달성하는 데 큰 도움을 준다는 것을 알 수 있었습니다. 더욱 복잡한 문제나 많은 데이터로 진행할수록 pre-trained 하는 부분이 중요하다는 것을 알 수 있었습니다.

### 위의 사항들을 구현하고 나온 결과들을 정리한 보고서를 README.md 형태로 업로드
### 코드 및 실행 결과는 jupyter notebook 형태로 같이 public github repository에 업로드하여 공유해주시면 됩니다(반드시 출력 결과가 남아있어야 합니다!!) 
