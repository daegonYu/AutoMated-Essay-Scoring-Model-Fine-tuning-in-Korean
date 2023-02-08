import torch, configargparse
from data import load_asap_data
from model_architechure_bert_multi_scale_multi_loss import DocumentBertScoringModel

import gc
import pandas as pd
from sklearn.model_selection import train_test_split

def _initialize_arguments(p: configargparse.ArgParser):
    p.add('--bert_model_path', help='bert_model_path')
    p.add('--efl_encode', action='store_true', help='is continue training')
    p.add('--r_dropout', help='r_dropout', type=float)
    p.add('--batch_size', help='batch_size', type=int)
    p.add('--bert_batch_size', help='bert_batch_size', type=int)
    p.add('--cuda', action='store_true', help='use gpu or not')
    p.add('--device')
    p.add('--model_directory', help='model_directory')
    p.add('--test_file', help='test data file')
    p.add('--data_dir', help='data directory to store asap experiment data')
    p.add('--data_sample_rate', help='data_sample_rate', type=float)
    p.add('--prompt', help='prompt')
    p.add('--fold', help='fold')
    p.add('--chunk_sizes', help='chunk_sizes', type=str)
    p.add('--result_file', help='pred result file path', type=str)

    args = p.parse_args()
    args.test_file = "%s/p8_fold3_test.txt" % args.data_dir
    args.model_directory = "%s/%s_%s" % (args.model_directory, args.prompt, args.fold)
    args.bert_model_path = args.model_directory

    print('--------------------------------')
    print('GPU 사용 여부 : ', torch.cuda.is_available())
    if torch.cuda.is_available() and args.cuda:
        args.device = 'cuda'
    else:
        args.device = 'cpu'
    return args


if __name__ == "__main__":

    gc.collect()
    torch.cuda.empty_cache()
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["asap.ini"])
    args = _initialize_arguments(p)
    print(args)

    # load train data
    essay_points = pd.read_csv('./datatouch/korproject/kor_essayset2_point.csv',index_col=0)
    logical_points = essay_points.논리성.to_list()
    novelty_points = essay_points.참신성.to_list()
    persuasive_points = essay_points.설득력.to_list()
    
    essays = pd.read_csv('./datatouch/korproject/kor_essayset2.csv', index_col=0)
    essays = essays.essay.to_list()
    
    tr_essay1, test_essay1, tr_logical_points, test_logical_points = train_test_split(essays, logical_points, test_size=0.2, random_state=321)
    tr_essay2, test_essay2, tr_novelty_points, test_novelty_points = train_test_split(essays, novelty_points, test_size=0.2, random_state=321)
    tr_essay3, test_essay3, tr_persuasive_points, test_persuasive_points = train_test_split(essays, persuasive_points, test_size=0.2, random_state=321)
    # tr_essay1 == tr_essay2 == tr_essay3 
    # test_essay1 == test_essay2 == test_essay3
    
    # if_sample=True
    # if if_sample:   # test data
    # train = load_asap_data('/home/daegon/Multi-Scale-BERT-AES/datatouch/prompt8_test.txt')
    # else:   # train data
    # train = load_asap_data('/home/daegon/Multi-Scale-BERT-AES/datatouch/prompt8_train.txt')

    # train_documents, train_labels = [], []        # 에세이별, 점수별
    # for _, text, label in train:
    #     train_documents.append(text)
    #     train_labels.append(label)
    
    # # load test data
    # # test = load_asap_data(args.test_file)
    # test = load_asap_data('/home/daegon/Multi-Scale-BERT-AES/datatouch/prompt8_test.txt')

    # test_documents, test_labels = [], []        # 에세이별, 점수별
    # for _, text, label in test:
    #     test_documents.append(text)
    #     test_labels.append(label)

    print("sample number:", len(essays))
    print("label number:", len(essay_points))

    # model1 = DocumentBertScoringModel(args=args)
    # model2 = DocumentBertScoringModel(args=args)
    # model3 = DocumentBertScoringModel(args=args)
    
    load_model = False        # 모델 불러오기
    
    config = './models/chunk_model.bin1/config.json'    # config는 모두 같다.
    chunk_model_path =  './models/chunk_model.bin1'; word_doc_model_path = './models/word_doc_model.bin1' 
    model1 = DocumentBertScoringModel(load_model=load_model,chunk_model_path=chunk_model_path,word_doc_model_path=word_doc_model_path,config=config,args=args)
    
    chunk_model_path =  './models/chunk_model.bin2'; word_doc_model_path = './models/word_doc_model.bin2'
    model2 = DocumentBertScoringModel(load_model=load_model,chunk_model_path=chunk_model_path,word_doc_model_path=word_doc_model_path,config=config,args=args)
    
    chunk_model_path =  './models/chunk_model.bin3'; word_doc_model_path = './models/word_doc_model.bin3'
    model3 = DocumentBertScoringModel(load_model=load_model,chunk_model_path=chunk_model_path,word_doc_model_path=word_doc_model_path,config=config,args=args)
    
    train_flag = True  
    if train_flag:
        model1.fit((tr_essay1, tr_logical_points))
        print('-'*20)
        print('model1 finish')
        print('-'*20)
        model2.fit((tr_essay1, tr_novelty_points))
        print('-'*20)
        print('model2 finish')
        print('-'*20)    
        model3.fit((tr_essay1, tr_persuasive_points))
        print('-'*20)
        print('model3 finish')
        print('-'*20)
        
    model1.predict_for_regress((test_essay1, test_logical_points))
    model2.predict_for_regress((test_essay1, test_novelty_points))
    model3.predict_for_regress((test_essay1, test_persuasive_points))

    # 예제넣고 결과 확인하기
    # input_sentence = [input()]      # list()와 []는 다르다.
    sentence = '인공지능은 처음부터 먼 길을 왔으며 오늘날 ChatGPT와 같은 AI 모델은 한때 불가능하다고 생각했던 작업을 수행하는 데 도움이 됩니다. OpenAI에서 개발한 ChatGPT는 인간과 같은 응답 생성, 질문 응답 및 텍스트 완성을 포함하여 광범위한 작업을 수행할 수 있는 언어 모델입니다. 인간 언어를 이해하고 생성하는 능력을 갖춘 ChatGPT는 우리가 기술과 상호 작용하는 방식을 혁신할 수 있는 잠재력을 가지고 있습니다. ChatGPT의 가장 흥미로운 점 중 하나는 광범위한 응용 프로그램에서 사용할 수 있는 잠재력입니다.'
    input_sentence = [sentence,'']      # list()와 []는 다르다. // 이중 []로 batch 표현
    
    mode_ = 'logical'
    model1.result_point(input_sentence =input_sentence ,mode_=mode_)
    
    mode_ = 'novelty'
    model2.result_point(input_sentence =input_sentence ,mode_=mode_)
    
    mode_ = 'persuasive'
    model3.result_point(input_sentence =input_sentence ,mode_=mode_)