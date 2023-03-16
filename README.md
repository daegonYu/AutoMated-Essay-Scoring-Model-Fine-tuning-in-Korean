# BERT를 활용한 에세이 점수 부여(AES) 모델 

### 모델이 에세이를 읽고 점수 산출

<br> 

## 모델 구조 
#### git : https://github.com/lingochamp/multi-scale-bert-aes 활용

<br>

# branch : korean_project
### <프로젝트 내용>
### Pretrained Model : KLUE-BERT (한국어 BERT) 사용
### Fine-tuning : 4가지 요소(논리성, 풍부한 근거, 설득력, 참신성)에 대해 평가된 1800개 한글 에세이 사용(Kaggle ASAP 데이터 -> 한글 번역 및 전처리 -> 라벨링)
### 손실 함수 계수 변경, 옵티마이저 변경, 교차 검증 등을 사용하여 모델 최적화 

<br>

# Repository : flakAESHomepage
### <프로젝트 내용>
### korean_project로 구현한 에세이 점수 산출 모델을 웹 페이지로 시각화 
### 기능 1. 다른 학생들의 점수와 4가지 요소에 대해 나의 점수 비교 

