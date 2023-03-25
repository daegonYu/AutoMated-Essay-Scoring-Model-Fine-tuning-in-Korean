# BERT를 활용한 에세이 점수 부여(AES) 모델 

### 모델이 에세이를 읽고 점수 산출하는 모델
논문 "On the Use of BERT for Automated Essay Scoring: Joint Learning of Multi-Scale Essay Representation" (NAACL 2022) 을 읽고 3가지 손실 함수 구현 및 Fine-tuning 코드 작성 <br>


<br> <br>

## 모델 구조 
위의 논문("On the Use of BERT for Automated Essay Scoring: Joint Learning of Multi-Scale Essay Representation") 모델 아키텍쳐 활용 (2022 SOTA) <br> Git : https://github.com/lingochamp/multi-scale-bert-aes

<br>

# 해당 Repository의 branch : korean_project 
### <프로젝트 내용>
- Pretrained Model : KLUE-BERT (한국어 BERT) 사용 <br>
- Fine-tuning : 4가지 요소(논리성, 풍부한 근거, 설득력, 참신성)에 대해 평가된 1800개 한글 에세이 사용(Kaggle ASAP 데이터 -> 한글 번역 및 전처리 -> 라벨링) <br>
- Fine-tuning 시 해당 데이터의 학습을 위해 손실 함수 계수 변경, 옵티마이저 변경, 교차 검증 등을 사용하여 모델 최적화  <br>
- 자세한 내용은 해당 Repository의 branch : korean_project의 README.md 에 적혀 있습니다.

<br>

# Repository : flakAESHomepage
### <프로젝트 내용>
1. korean_project로 구현한 4가지 요소에 대해 에세이 점수 산출 모델을 웹 페이지로 시각화 <br>
2. 자신의 에세이가 어떠한 지 다른 학생과 비교해서 점수와 상위 %로 알려줌으로써 자신의 글쓰기가 어떠한 지 보다 쉽게 알 수 있게 한 것이 특징 <br>
3. 자신의 글에서 중 부족한 요소들을 하단에 Tip을 적어놔서 부족한 요소에 대해 다음에 좀 더 높은 점수를 받을 수 있게 도와줌 <br>
- 자세한 내용은 Repository : flakAESHomepage의 README.md 에 적혀 있습니다.
