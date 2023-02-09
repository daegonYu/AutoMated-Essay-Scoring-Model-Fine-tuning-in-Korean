import os
import torch
from transformers import BertConfig, CONFIG_NAME, BertTokenizer
from transformers import AutoModel, AutoTokenizer
from document_bert_architectures import DocumentBertCombineWordDocumentLinear, DocumentBertSentenceChunkAttentionLSTM
from evaluate import evaluation
from encoder import encode_documents
from data import asap_essay_lengths, fix_score
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import os.path


def sim(y,yhat):
    e = torch.tensor(1e-8)
    m = torch.pow(torch.pow(y,2).sum(),0.5) * torch.pow(torch.pow(yhat,2).sum(),0.5)
    similarity = torch.sum(y*yhat) / torch.max(torch.tensor([m,e]))
    # cos = nn.CosineSimilarity(dim=0, eps=1e-8)      # batchsize만큼의 점수리스트가 들어옴으로 dim=0
    # loss = 1- cos(y,yhat)
    loss = 1 - similarity
    return loss            # 논문보니 평균내지 않는다.

def mr_loss_func(pred,label):
    # 배치사이즈가 1이 되면 에러가 나온다.
    mr_loss = 0
    for i in range(pred.size(0)):
        y = pred - pred[i]
        yhat = label - label[i]
        yhat = yhat.sign()
        mask = y.sign() != yhat.sign()

        mr_loss += y[mask].abs().sum()
        
    return mr_loss/label.size(0)

class DocumentBertScoringModel():
    def __init__(self, load_model, chunk_model_path,word_doc_model_path,config, args=None):
        if args is not None:
            self.args = vars(args)
        # self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])       # 토크나이저는 vacob.txt 파일 기준으로
        # 토크나이저
        # self.bert_tokenizer = BertTokenizer.from_pretrained('klue/bert-base')
        self.bert_tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")   # transformer = 4.7.0 

        # config설정
        # if os.path.exists(self.args['bert_model_path']):
        #     if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
        #         config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
        #     elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
        #         config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
        #     else:
        #         raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        # else:
        #     config = BertConfig.from_pretrained(self.args['bert_model_path'])
            # config는 제외하자.
        self.config = config
        self.prompt = int(args.prompt[1])
        chunk_sizes_str = self.args['chunk_sizes']
        self.chunk_sizes = []
        self.bert_batch_sizes = []
        if "0" != chunk_sizes_str:
            for chunk_size_str in chunk_sizes_str.split("_"):
                chunk_size = int(chunk_size_str)
                self.chunk_sizes.append(chunk_size)
                bert_batch_size = int(asap_essay_lengths[self.prompt] / chunk_size) + 1
                self.bert_batch_sizes.append(bert_batch_size)
        bert_batch_size_str = ",".join([str(item) for item in self.bert_batch_sizes])

        print("prompt:%d, asap_essay_length:%d" % (self.prompt, asap_essay_lengths[self.prompt]))
        print("chunk_sizes_str:%s, bert_batch_size_str:%s" % (chunk_sizes_str, bert_batch_size_str))
        

        # 저장된 파라미터 불러오기 => load_model
        if load_model:
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                word_doc_model_path,
                config=config)
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                chunk_model_path,
                config=config)
            
            # 추가학습시키는 거 아니면 eval 모드로 변경해두자.
            # self.bert_regression_by_word_document.eval()
            # self.bert_regression_by_chunk.eval()
            
        # 초기화된 모델 사용하기
        else:
            # bert-base-uncased를 사용하는데 초기화되지 않은 변수들은 자동 초기화 시켜줌
            # 초기화된 파라미터를 사용하므로 다음과 같은 메세지가 뜬다.
            # You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
            # => 구체적으로 풀고 싶은 문제에 대해 모델을 학습시켜라(Fine-Tuning), 예측이나 추론이 가능하게
            # from_pretrained()이 인수로는 (pretrained_model_name_or_path, config)가 있는데 pretrained된 모델 선택과 config에 계층에 대한 정보를 넣어줘야 한다.
            # 둘다 pretrain 모델로 bert-base-uncased가 맞는 것 같다. longformer를 사용하면 초기화되는 파라미터들이 너무 많아서 그런지 학습 진행이 안된다.
            self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
                "klue/bert-base",
            )
            self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
                "klue/bert-base"
            )
            # 아래 원본
            # self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
            #     self.args['bert_model_path'] + "/word_document",
            #     config=config
            # )
            # self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
            #     self.args['bert_model_path'] + "/chunk",
            #     config=config)
        

    def predict_for_regress(self, data):    # data = 한글에세이, 정답
        # test 데이터가 들어가서 eval 결과 확인하기
        correct_output = None
        if isinstance(data, tuple) and len(data) == 2:
            # 토크나이징
            document_representations_word_document, document_sequence_lengths_word_document = encode_documents(
                data[0], self.bert_tokenizer, max_input_length=512)
            document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []
            for i in range(len(self.chunk_sizes)):
                document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                    data[0],
                    self.bert_tokenizer,
                    max_input_length=self.chunk_sizes[i])
                document_representations_chunk_list.append(document_representations_chunk)
                document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)
            correct_output = torch.FloatTensor(data[1])     # data[1]에는 정답이 들어있다.

        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])

        self.bert_regression_by_word_document.eval()    # eval 모드로 변경
        self.bert_regression_by_chunk.eval()

        with torch.no_grad():       
            predictions = torch.empty((document_representations_word_document.shape[0]))
            for i in range(0, document_representations_word_document.shape[0], self.args['batch_size']):    # 1 iteration
                batch_document_tensors_word_document = document_representations_word_document[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_predictions_word_document = self.bert_regression_by_word_document(batch_document_tensors_word_document, device=self.args['device'])
                batch_predictions_word_document = torch.squeeze(batch_predictions_word_document)

                batch_predictions_word_chunk_sentence_doc = batch_predictions_word_document
                for chunk_index in range(len(self.chunk_sizes)):
                    batch_document_tensors_chunk = document_representations_chunk_list[chunk_index][i:i + self.args['batch_size']].to(
                        device=self.args['device'])
                    batch_predictions_chunk = self.bert_regression_by_chunk(
                        batch_document_tensors_chunk,
                        device=self.args['device'],
                        bert_batch_size=self.bert_batch_sizes[chunk_index]
                    )
                    batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                    batch_predictions_word_chunk_sentence_doc = torch.add(batch_predictions_word_chunk_sentence_doc, batch_predictions_chunk)
                predictions[i:i + self.args['batch_size']] = batch_predictions_word_chunk_sentence_doc
        assert correct_output.shape == predictions.shape 

        prediction_scores = []
        label_scores = []
        predictions = predictions.cpu().numpy()
        correct_output = correct_output.cpu().numpy()
        outfile = open(os.path.join(self.args['model_directory'], self.args['result_file']), "w")
        for index, item in enumerate(predictions):
            prediction_scores.append(fix_score(item, self.prompt))
            label_scores.append(correct_output[index])
            outfile.write("%f\t%f\n" % (label_scores[-1], prediction_scores[-1]))
        outfile.close()

        test_eva_res = evaluation(label_scores, prediction_scores)
        print("pearson:", float(test_eva_res[7]))
        print("qwk:", float(test_eva_res[8]))
        
        # txt 파일에 이어쓰기
        f = open('./loss_eval/eval.txt','a')
        f.write("\npearson: {} \t qwk: {}".format(float(test_eva_res[7]),float(test_eva_res[8])))
        f.close()
        
        return float(test_eva_res[7]), float(test_eva_res[8])

    def fit(self, data):    # 학습하는 부분 (학습데이터)
        lr = 6e-5
        # epoch 1/4 해서 실험   epoch 20 실험함
        epochs = 20     # 80
        weight_decay = 0.005    # 논문 : 0.005
        word_document_optimizer = torch.optim.Adam(self.bert_regression_by_word_document.parameters(),lr=lr,weight_decay=weight_decay)
        chunk_optimizer = torch.optim.Adam(self.bert_regression_by_chunk.parameters(),lr=lr,weight_decay=weight_decay)
        
        word_document_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=word_document_optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
        chunk_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=chunk_optimizer,
                                        lr_lambda=lambda epoch: 0.95 ** epoch)
        
        correct_output = None
        if isinstance(data, tuple) and len(data) == 2:
            document_representations_word_document, document_sequence_lengths_word_document = encode_documents(
                data[0], self.bert_tokenizer, max_input_length=512)
            document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []
            for i in range(len(self.chunk_sizes)):
                document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                    data[0],
                    self.bert_tokenizer,
                    max_input_length=self.chunk_sizes[i])
                document_representations_chunk_list.append(document_representations_chunk)
                document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)
            correct_output = torch.FloatTensor(data[1])     # data[1]에는 정답이 들어있다.

        # 모델 device에 통일
        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])
        
        self.bert_regression_by_word_document.zero_grad()   # model gradient 초기화
        self.bert_regression_by_chunk.zero_grad()
        
        self.bert_regression_by_word_document.train()   # train 모드로 변경
        self.bert_regression_by_chunk.train()
        loss_list = []
        for epoch in tqdm(range(epochs)):
            for i in tqdm(range(0, document_representations_word_document.shape[0], self.args['batch_size'])):    # iteration
                # 배치마다 device를 통일 시켜줘야 한다.
                batch_document_tensors_word_document = document_representations_word_document[i:i + self.args['batch_size']].to(device=self.args['device'])
                batch_predictions_word_document = self.bert_regression_by_word_document(batch_document_tensors_word_document, device=self.args['device'])
                batch_predictions_word_document = torch.squeeze(batch_predictions_word_document)

                batch_predictions_word_chunk_sentence_doc = batch_predictions_word_document
                for chunk_index in range(len(self.chunk_sizes)):
                    batch_document_tensors_chunk = document_representations_chunk_list[chunk_index][i:i + self.args['batch_size']].to(
                        device=self.args['device'])
                    batch_predictions_chunk = self.bert_regression_by_chunk(
                        batch_document_tensors_chunk,
                        device=self.args['device'],
                        bert_batch_size=self.bert_batch_sizes[chunk_index]
                    )
                    batch_predictions_chunk = torch.squeeze(batch_predictions_chunk)
                    batch_predictions_word_chunk_sentence_doc = torch.add(batch_predictions_word_chunk_sentence_doc, batch_predictions_chunk)
                
                # F를 사용한 loss function은 평균 내서 나온다.
                # 배치마다 device를 통일 시켜줘야 한다.
                mse_loss = F.mse_loss(batch_predictions_word_chunk_sentence_doc,correct_output[i:i + self.args['batch_size']].to(device=self.args['device']))  # 평균되어서 나온다.
                sim_loss = sim(batch_predictions_word_chunk_sentence_doc,correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])) 
                mr_loss = mr_loss_func(batch_predictions_word_chunk_sentence_doc, correct_output[i:i + self.args['batch_size']].to(device=self.args['device'])) # 평균되어서 나온다.
                a=2;b=1;c=1
                total_loss = a*mse_loss + b*sim_loss + c*mr_loss
                print('Epoch : {}, iter: {}, Loss : {}'.format(epoch+1, i+1, total_loss.item()))
                loss_list.append(total_loss.item())
                
                total_loss.backward()   # 기울기 계산
                
                word_document_optimizer.step()  # 파라미터 갱신
                chunk_optimizer.step()
                
                word_document_optimizer.zero_grad() # 기울기 초기화
                chunk_optimizer.zero_grad()
            
                
            word_document_scheduler.step()  # 학습률 업데이트
            chunk_scheduler.step()
        
        # 모델 저장 nn.Module 사용시로 알고 있다.
        model_save = False
        # _PATH = '/home/daegon/Multi-Scale-BERT-AES/models'
        # if model_save:
        #     torch.save(self.bert_regression_by_word_document.state_dict(), _PATH+'/word_doc.pt')
        #     torch.save(self.bert_regression_by_chunk.state_dict(), _PATH+'/chunk.pt')
        #     print('success the model save')
        
        # 손실그래프 및 손실 값 확인하기
        graph = True
        if graph:
            plt.plot(range(len(loss_list)),loss_list)
            plt.show()
            loss_list = np.array(loss_list)
            np.save('./loss_eval/loss_1.npy',loss_list)
            
        # pretrained 모델 저장하기
        _save = True
        for i in range(1,100):
            if os.path.exists('./models/word_doc_model.bin{}'.format(i)):
                continue
            else :
                self.bert_regression_by_word_document.save_pretrained('./models/word_doc_model.bin{}'.format(i))
                self.bert_regression_by_chunk.save_pretrained('./models/chunk_model.bin{}'.format(i))
                break
     
    def result_point(self, input_sentence, mode_):    # 예제 넣어보기
        # correct_output = None
        # 토크나이징
        document_representations_word_document, document_sequence_lengths_word_document = encode_documents(
            input_sentence, self.bert_tokenizer, max_input_length=512)
        
        document_representations_chunk_list, document_sequence_lengths_chunk_list = [], []
        
        for i in range(len(self.chunk_sizes)):
            document_representations_chunk, document_sequence_lengths_chunk = encode_documents(
                input_sentence,
                self.bert_tokenizer,
                max_input_length=self.chunk_sizes[i])   # 맥스길이를 chunk size로 설정
            document_representations_chunk_list.append(document_representations_chunk)
            document_sequence_lengths_chunk_list.append(document_sequence_lengths_chunk)    # 토크나이즈한거 다 리스트에 추가.
        # correct_output = torch.FloatTensor(data[1])     # data[1]에는 정답이 들어있다.

        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])

        self.bert_regression_by_word_document.eval()    # eval 모드로 변경
        self.bert_regression_by_chunk.eval()

        with torch.no_grad():           # 기울기 저장 X
            # predictions = torch.empty((document_representations_word_document.shape[0]))
            # 한 문장 삽입
            document_tensors_word_document = document_representations_word_document[0:0+1].to(device=self.args['device'])
            # 토크나이즈 한 것을 모델에 삽입
            predictions_word_document = self.bert_regression_by_word_document(document_tensors_word_document, device=self.args['device'])
            predictions_word_document = torch.squeeze(predictions_word_document)

            predictions_word_chunk_sentence_doc = predictions_word_document
            for chunk_index in range(len(self.chunk_sizes)):
                document_tensors_chunk = document_representations_chunk_list[chunk_index][0:0+1].to(
                    device=self.args['device'])
                predictions_chunk = self.bert_regression_by_chunk(
                    document_tensors_chunk,
                    device=self.args['device'],
                    bert_batch_size=self.bert_batch_sizes[chunk_index]
                )
                predictions_chunk = torch.squeeze(predictions_chunk)
                predictions_word_chunk_sentence_doc = torch.add(predictions_word_chunk_sentence_doc, predictions_chunk)
            # predictions[0] = predictions_word_chunk_sentence_doc
            
            pred_point = float(predictions_word_chunk_sentence_doc)
            # pred_point range : 0~5
            if pred_point < 0:
                pred_point = 0
            elif pred_point > 5:
                pred_point = 5
            pred_point  *= 20
            pred_point = round(pred_point,2)
                
        if mode_ == 'logical':
            print("{} 예측 점수 : {}점".format('논리성',pred_point))
        
        elif mode_ == 'novelty':
            print("{} 예측 점수 : {}점".format('참신성',pred_point))
        
        elif mode_ == 'persuasive':
            print("{} 예측 점수 : {}점".format('설득력',pred_point))
            
        else:
            print("{} 예측 점수 : {}점".format('?',pred_point))
        
        return pred_point
    
