# 자동 에세이 채점(Automated Essay Scoring; AES) 모델 

## Setting

1. python=3.6+
2. pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -Ur requirements.txt     // 실행하면 오류없이 환경설정 끝
3. pip install azureml-sdk[automl,explain,notebooks]>=1.42.0  // protoc 오류 뜰때
4. data 경로: /home/daegon/Multi-Scale-BERT-AES/data 로 해주고 p8_fold3_test.txt라는 이름이 데이터파일 이어야함
5. model 경로: model_directory: /home/daegon/Multi-Scale-BERT-AES/data 이렇게 해두고 밑에 p8_3 폴더가 있어야함 (p8_3 폴더는 https://pan.baidu.com/s/1_m_-DQlX-dLh1XdhOMzj1A?pwd=tmmb  에서 다운 받은 것)


home='current_path' <br>
bert_model_path="${home}/Multi-Scale-BERT-AES/data/p8_3"<br>
data_dir="${home}/Multi-Scale-BERT-AES/data"<br>
model_directory="${home}/Multi-Scale-BERT-AES/data/p8_3"<br>
result_file="${home}/Multi-Scale-BERT-AES/result.txt"<br>
test_file="${home}/Multi-Scale-BERT-AES/data/p8_fold3_test.txt"


## 프로젝트 내용
- Pretrained Model : KLUE-BERT (한국어 BERT) 사용 <br>
- Model Architecture : Paper "On the Use of BERT for Automated Essay Scoring: Joint Learning of Multi-Scale Essay Representation" <br>
## Fine-tuning 
- Data set : 4가지 요소(논리성, 풍부한 근거, 설득력, 참신성)에 대해 평가된 1800개 한글 에세이 사용(Kaggle ASAP 데이터 -> 한글 번역 및 전처리 -> 라벨링) <br>
### 아래 3가지에 대해 모델 최적화 진행
- 손실 함수 <br> 
- 옵티마이저 <br> 
- 교차 검증 <br>

More details, 해당 Repository의 branch : korean_project의 README.md

<br>

## Repository : flakAESHomepage
### 프로젝트 내용
1. korean_project로 구현한 4가지 요소에 대해 에세이 점수 산출 모델을 웹 페이지로 시각화 <br>
2. 자신의 에세이가 어떠한 지 다른 학생과 비교해서 점수와 상위 %로 알려줌으로써 자신의 글쓰기가 어떠한 지 보다 쉽게 알 수 있게 한 것이 특징 <br>
3. 자신의 글에서 중 부족한 요소들을 하단에 Tip을 적어놔서 부족한 요소에 대해 다음에 좀 더 높은 점수를 받을 수 있게 도와줌 <br>

More details, Repository : flakAESHomepage의 README.md 
