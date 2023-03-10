import os
import torch
from transformers import BertConfig, CONFIG_NAME, BertTokenizer
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
def sim(y,yhat):
    e = torch.tensor(1e-8)
    m = torch.pow(torch.pow(y,2).sum(),0.5) * torch.pow(torch.pow(yhat,2).sum(),0.5)
    similarity = torch.sum(y*yhat) / torch.max(m,e)
    return 1- similarity


class DocumentBertScoringModel():
    def __init__(self, args=None):
        if args is not None:
            self.args = vars(args)
        self.bert_tokenizer = BertTokenizer.from_pretrained(self.args['bert_model_path'])
        if os.path.exists(self.args['bert_model_path']):
            if os.path.exists(os.path.join(self.args['bert_model_path'], CONFIG_NAME)):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], CONFIG_NAME))
            elif os.path.exists(os.path.join(self.args['bert_model_path'], 'bert_config.json')):
                config = BertConfig.from_json_file(os.path.join(self.args['bert_model_path'], 'bert_config.json'))
            else:
                raise ValueError("Cannot find a configuration for the BERT based model you are attempting to load.")
        else:
            config = BertConfig.from_pretrained(self.args['bert_model_path'])
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
        self.bert_regression_by_word_document = DocumentBertCombineWordDocumentLinear.from_pretrained(
            self.args['bert_model_path'] + "/word_document",
            config=config
        )
        self.bert_regression_by_chunk = DocumentBertSentenceChunkAttentionLSTM.from_pretrained(
            self.args['bert_model_path'] + "/chunk",
            config=config)

    def predict_for_regress(self, data):
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

        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])

        self.bert_regression_by_word_document.eval()    # eval 모드로 변경
        self.bert_regression_by_chunk.eval()

        with torch.no_grad():       
            predictions = torch.empty((document_representations_word_document.shape[0]))
            for i in range(0, document_representations_word_document.shape[0], self.args['batch_size']):
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
        return float(test_eva_res[7]), float(test_eva_res[8])

    def fit(self, data):
        word_document_optimizer = torch.optim.Adam(self.bert_regression_by_word_document.parameters(),lr=0.01)
        chunk_optimizer = torch.optim.Adam(self.bert_regression_by_chunk.parameters(),lr=0.01)
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

        self.bert_regression_by_word_document.to(device=self.args['device'])
        self.bert_regression_by_chunk.to(device=self.args['device'])

        self.bert_regression_by_word_document.train()
        self.bert_regression_by_chunk.train()
   
        predictions = torch.empty((document_representations_word_document.shape[0]))
        for i in range(0, document_representations_word_document.shape[0], self.args['batch_size']):    # iteration
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
                # print(batch_predictions_word_chunk_sentence_doc.shape,correct_output[i:i + self.args['batch_size']].shape)
            mse_loss = F.mse_loss(batch_predictions_word_chunk_sentence_doc,correct_output[i:i + self.args['batch_size']])
            sim_loss = sim(batch_predictions_word_chunk_sentence_doc,correct_output[i:i + self.args['batch_size']])
            mr_loss = F.margin_ranking_loss(batch_predictions_word_chunk_sentence_doc,correct_output[i:i + self.args['batch_size']],batch_predictions_word_chunk_sentence_doc.sign())
            total_loss = (mse_loss + sim_loss + mr_loss) / correct_output[i:i + self.args['batch_size']].size(0)
            total_loss.backward()
            word_document_optimizer.step()
            chunk_optimizer.step()
            word_document_optimizer.zero_grad()
            chunk_optimizer.zero_grad()
                
            predictions[i:i + self.args['batch_size']] = batch_predictions_word_chunk_sentence_doc  # prediction에는 모든 y값(추정값)이 저장된다.
            
        assert correct_output.shape == predictions.shape 

        prediction_scores = []
        label_scores = []
        predictions = predictions.cpu().numpy()
        correct_output = correct_output.cpu().numpy()
        outfile = open(os.path.join(self.args['model_directory'], self.args['result_file']), "w")   # 결과 저장
        for index, item in enumerate(predictions):
            prediction_scores.append(fix_score(item, self.prompt))      # self.propt = 8
            label_scores.append(correct_output[index])
            outfile.write("%f\t%f\n" % (label_scores[-1], prediction_scores[-1]))
        outfile.close()     # 결과저장

        test_eva_res = evaluation(label_scores, prediction_scores)  # pearson, qwk 계산
        print("pearson:", float(test_eva_res[7]))
        print("qwk:", float(test_eva_res[8]))
        return float(test_eva_res[7]), float(test_eva_res[8])
