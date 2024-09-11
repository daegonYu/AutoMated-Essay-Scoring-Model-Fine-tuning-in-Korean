# AES(Automated Essay Scoring) 모델 

## Setting

1. python=3.6+
2. pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -Ur requirements.txt     // 실행하면 오류없이 환경설정 끝
3. pip install azureml-sdk[automl,explain,notebooks]>=1.42.0  // protoc 오류 뜰때
4. data 경로: /home/daegon/Multi-Scale-BERT-AES/data 로 해주고 p8_fold3_test.txt라는 이름이 데이터파일 이어야함
5. model 경로: model_directory: /home/daegon/Multi-Scale-BERT-AES/data 이렇게 해두고 밑에 p8_3 폴더가 있어야함 (p8_3 폴더는 https://pan.baidu.com/s/1_m_-DQlX-dLh1XdhOMzj1A?pwd=tmmb  에서 다운 받은 것)


home='current_path'<br>
bert_model_path="${home}/Multi-Scale-BERT-AES/data/p8_3"<br>
data_dir="${home}/Multi-Scale-BERT-AES/data"<br>
model_directory="${home}/Multi-Scale-BERT-AES/data/p8_3"<br>
result_file="${home}/Multi-Scale-BERT-AES/result.txt"<br>
test_file="${home}/Multi-Scale-BERT-AES/data/p8_fold3_test.txt"<br>


## 대조학습을 이용한 Loss 추가
- 두 에세이의 점수 차가 작다면 두 에세이의 코사인 유사도도 작아야 한다는 가설로 코드 작성
- 해당 손실 함수는 cl_loss_func() 라는 함수로 구현되어 있음
