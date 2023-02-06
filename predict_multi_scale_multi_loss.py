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
        args.dev = 'cpu'
    return args


if __name__ == "__main__":

    gc.collect()
    torch.cuda.empty_cache()
    # initialize arguments
    p = configargparse.ArgParser(default_config_files=["asap.ini"])
    args = _initialize_arguments(p)
    print(args)

    # load train data
    essay_points = pd.read_csv('/home/daegon/Multi-Scale-BERT-AES/datatouch/korproject/kor_essayset2_point.csv',index_col=0)
    logical_points = essay_points.논리성.to_list()
    novelty_points = essay_points.참신성.to_list()
    persuasive_points = essay_points.설득력.to_list()
    
    essays = pd.read_csv('/home/daegon/Multi-Scale-BERT-AES/datatouch/korproject/kor_essayset2.csv', index_col=0)
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

    model1 = DocumentBertScoringModel(args=args)
    model2 = DocumentBertScoringModel(args=args)
    model3 = DocumentBertScoringModel(args=args)
    
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
