# 4가지 요소에 대한 글쓰기 점수 평가 모델
### 4가지 요소 : 논리성, 근거의 풍부함, 설득력, 참신성

<br>

## 모델 구조 

Git : https://github.com/lingochamp/multi-scale-bert-aes <br>
Kaggle ASAP데이터에서 가장 높은 성능을 내는 모델 중 하나 <br>

## 사전 학습 모델
Pretrained Model : KLUE-BERT <br>
한국어로 사전 학습된 BERT 모델 중 하나

<br>

## 사용한 데이터셋
연구과제로 직접 만든 데이터셋 사용 <br>
Kaggle : ASAP 에세이 데이터셋의 Prompt 2를 한국어 번역 및 전처리(번역투 수정, 어색한 문장 삭제) 작업한 후 6명의 연구원이 한 에세이에 대해 9가지 요소에 대해 점수를 부여함 <br>
9가지 요소 중 4가지 요소만 사용하여 모델 학습 <br>
데이터의 개수 : 1800 개

<br>

## 최종 모델의 성능

#### 논리성 모델 
pearson : 0.869 qwk : 0.82 <br>


#### 근거의 풍부함 모델 <br>
pearson: 0.894 qwk: 0.81 <br>


#### 설득력 모델 <br>
pearson : pearson: 0.899 qwk: 0.795 <br>


#### 참신성 모델 <br>
pearson: 0.743 qwk: 0.708 <br>


## 모델의 성능 비교

#### <손실함수 설명>

MSE(Mean Square Error) : 예측 값과 정답 값의 차이만큼 손실 발생 <br>
SIM 손실 함수 : 예측 값과 정답 값의 코사인 유사도를 계산 <br>
MR(Margin Ranking) : 배치 데이터에서 i번째 정답 값이 j번째 정답 값보다 크다면 예측 값도 마찬가지로 i번째 예측 값이 j번째 예측 값보다 커야함, 하지만 같거나 작으면 예측 값의 차이 만큼 손실 발생

<br>

-----


옵티마이져는 가장 대표적인 Adam 사용 <br>
예측 점수와 정답 간의 점수 차를 줄이는 것이 중요하다고 생각하여 MSE의 손실 함수 가중치 2로 설정 <br>

<br>

Optimizer : Adam <br>
손실함수 계수 : MSE, SIM, MR = 2, 1, 1 <br>

논리성 모델       pearson: 0.529 	 qwk: 0.431 <br>
근거의 풍부함 모델 pearson: 0.585 	 qwk: 0.583 <br>
설득력 모델       pearson: 0.516 	 qwk: 0.466 <br>
참신성 모델       pearson: 0.459 	 qwk: 0.431 <br>

-----

하나의 모델을 학습시키는데 걸리는 시간 5~6시간이기 때문에 모든 요소가 아닌 하나의 요소만 선택하여 실험 <br>


lr 스케줄러 추가 <br>
lr 스케줄러는 학습 시 학습률(lr)를 조금씩 줄여줌으로써 Global Minimum에 도달하도록 학습을 도와주는 역할을 함 <br>

Opimizer : Adam <br>
lr 스케줄러 : LambdaLR <br>
손실함수 계수 : MSE, SIM, MR = 2, 1, 1 <br>

근거의 풍부함 모델  pearson:0.577, qwk:0.550 <br>

-----------

옵티마이저 RAdam으로 변경 (RAdam : Rectified Adam)으로 <br>
Adam의 단점을 보안한 옵티마이저


Opimizer : RAdam <br>
손실함수 계수 : MSE, SIM, MR = 2, 1, 1 <br>

근거의 풍부함 모델  pearson:0.644, qwk:0.547 <br>


-----

손실 함수 계수 변경 및 Adam + lr 스케줄러와 RAdam과의 성능 비교 <br>


Opimizer : Adam <br>
lr 스케줄러 : LambdaLR <br>
손실함수 계수 : MSE, SIM, MR = 3, 0, 1 <br>

근거의 풍부함 모델 pearson:0.557, qwk:0.526 <br>


----

Opimizer : RAdam <br>
손실함수 계수 : MSE, SIM, MR = 3, 0, 1 <br>

근거의 풍부함 모델 pearson:0.553, qwk:0.470 <br>


------

데이터가 1800개 밖에 안되므로 모든 데이터셋을 훈련에 활용하여 고정된 train set과 test set의 과적합을 막고 일반화된 모델 생성을 위해 교차 검증 시행

5-fold validation <br>
Opimizer : RAdam <br>
손실함수 계수 : MSE, SIM, MR = 3, 0, 1 <br>

근거의 풍부함 모델 pearson: 0.786 qwk: 0.679 <br>


-----------

pearson 계수와 SIM 손실 함수와 연관 있다고 생각하여 SIM 손실 함수 추가 <br>


5-fold validation <br>
Opimizer : RAdam <br>
손실함수 계수 : MSE, SIM, MR = 3, 1, 2 <br>

근거의 풍부함 모델 pearson: 0.908 qwk: 0.847 <br>


------
### 최종 학습 



5-fold validation <br>
Opimizer : RAdam <br>
손실함수 계수 : MSE, SIM, MR = 3, 1, 2 <br>


논리성 모델      pearson : 0.869 qwk : 0.821 <br>
근거의 풍부함 모델 pearson: 0.894 qwk: 0.81 <br>
설득력 모델       pearson : pearson: 0.899 qwk: 0.795 <br>
참신성 모델       pearson: 0.743 qwk: 0.708  <br>



