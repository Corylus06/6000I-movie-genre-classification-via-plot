# Bert do multi-label text classification

import nltk
import keras
import pandas as pd
import tensorflow as tf
import numpy as np
import re
import random
import transformers
import os, json, gc, re, random
from tqdm.notebook import tqdm
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import tokenizers
from tqdm import tqdm_notebook, trange
from keras import optimizers
from keras import layers
from keras import Input
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.nn import CrossEntropyLoss,BCEWithLogitsLoss
from sklearn.metrics import multilabel_confusion_matrix


def load_one_hot_data():
    movies_df = pd.read_csv("./drive/MyDrive/movies_preprocessed5.csv")
    movies_df = movies_df.rename(columns = {"PlotClean": "Plot"})
    # movies_df["genre_encoded"] = movies_df['action']+movies_df['animation']+movies_df['comedy']+movies_df['crime']+movies_df['drama']+movies_df['musical']+movies_df['romance']+movies_df['thriller']
    movies_label = movies_df[["action", "animation", "comedy", "crime", "drama", "musical", "romance", "thriller"]]
    id = movies_df['id']
    id_list = id.values.tolist()
    # print(id_list)
    # print(movies_label)
    # print(movies_df)
    label = movies_label.values.tolist()
    label_dataframe = pd.DataFrame({'id':id_list, 'label':label})
    # print(label_dataframe)
    movies_df = pd.merge(movies_df, label_dataframe, on='id')
    # print(label)
    # print(movies_df)
    movies_df = movies_df[["Plot", "label"]]
    label_encoder = LabelEncoder()
    movies_df["genre_encoded"] = label_encoder.fit_transform(movies_df["label"].map(str).tolist())
    class_nums = movies_df["genre_encoded"].max() + 1
    # print(class_nums)
    # print(label_encoder.classes_)
    movies_df = movies_df.groupby("genre_encoded").head(600).reset_index(drop=True)
    return movies_df


def Bert_preprocess(movies_data):
    max_length = 512
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sequences = list(movies_data['Plot'])
    labels = list(movies_data['label'])
    # print("labels")
    # print(labels)
    masks = []
    segments = []
    for i in range(len(sequences)):
      if len(sequences[i]) > (max_length - 2):
        sequences[i] = sequences[i][0:(max_length - 2)]
      sequences[i] = '[CLS]' + sequences[i] + '[SEP]'
      sequences[i] = bert_tokenizer.tokenize(sequences[i])
      sequences[i] = bert_tokenizer.convert_tokens_to_ids(sequences[i])
      padding = [0] * (max_length - len(sequences[i]))
      seq_mask = [1] * len(sequences[i]) + padding
      seq_segment = [0] * len(sequences[i]) + padding
      sequences[i] = sequences[i] + padding
      masks.append(seq_mask)
      segments.append(seq_segment)

    # print(sequences[0])
    # print(masks[0])
    # print(segments[0])
    return sequences, masks, segments, labels


def create_model():
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=8)
    return model


# multi-label with train and test
if __name__ == "__main__":
    nltk.download('stopwords')
    movies_data = load_one_hot_data()
    batch_size = 4
    epoches = 2
    print(movies_data)
    train_data, test_data = train_test_split(movies_data, test_size=0.1)
    genre_numes = 8

    data, data_mask, data_segment, train_labels = Bert_preprocess(train_data)
    test_data, test_data_mask, test_data_segment, test_labels = Bert_preprocess(test_data)

    # train_data
    t_seqs = torch.tensor(data, dtype=torch.long)
    t_seq_masks = torch.tensor(data_mask, dtype=torch.long)
    t_seq_segments = torch.tensor(data_segment, dtype=torch.long)
    t_labels = torch.tensor(train_labels, dtype=torch.float)
    train_data = TensorDataset(t_seqs, t_seq_masks, t_seq_segments, t_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloder = DataLoader(dataset=train_data, sampler=train_sampler, batch_size=4)
    device = 'cuda'
    model = create_model()
    model.to(device)
    model.train()
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params':
                [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
                0.01
        },
        {
            'params':
                [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
                0.0
        }
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=2e-05,
                         warmup=0.1,
                         t_total=(epoches * len(data) / batch_size) + 1)

    for i in trange(epoches, desc='Epoch'):
        loss_collect = []
        for step, batch_data in enumerate(tqdm_notebook(train_dataloder, desc='Iteration')):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            logits = model(batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            # softmax no needed since CrossEntropyloss has the softmax
            # logits = logits.sigmoid(dim=1)
            # print(logits)
            # print(batch_labels)
            loss_function = BCEWithLogitsLoss()
            loss = loss_function(logits, batch_labels)
            loss.backward()
            loss_collect.append(loss.item())
            # print(loss.item())
            optimizer.step()
            optimizer.zero_grad()
            # plt.figure(figsize=(12,8))
            # plt.plot(range(len(loss_collect)), loss_collect,'g.')
            # plt.grid(True)
            # plt.show()
        sum_loss = np.sum(loss_collect)
        print("average_loss:", sum_loss / len(loss_collect))

    # test_data
    test_seqs = torch.tensor(test_data, dtype=torch.long)
    test_seq_masks = torch.tensor(test_data_mask, dtype=torch.long)
    test_seq_segments = torch.tensor(test_data_segment, dtype=torch.long)
    test_labels = torch.tensor(test_labels, dtype=torch.float)
    test_data = TensorDataset(test_seqs, test_seq_masks, test_seq_segments, test_labels)
    test_dataloder = DataLoader(dataset=test_data, batch_size=4)

    preds = []
    test_label = []
    threshold = 0.5
    model.eval()
    with torch.no_grad():
        for batch_data in tqdm_notebook(test_dataloder, desc='TEST'):
            batch_data = tuple(t.to(device) for t in batch_data)
            batch_seqs, batch_seq_masks, batch_seq_segments, batch_labels = batch_data
            logits = model(batch_seqs, batch_seq_masks, batch_seq_segments, labels=None)
            logits = logits.sigmoid()
            logits = logits.cpu().detach().numpy()
            logits[logits >= threshold] = 1
            logits[logits < threshold] = 0
            batch_labels = batch_labels.cpu().detach().numpy()
            for i in range(len(logits)):
                preds.append(logits[i])
                test_label.append(batch_labels[i])
    # print(preds)
    # print(test_label)
    multi_label_confusion_matrix = multilabel_confusion_matrix(test_label, preds)
    # print(multi_label_confusion_matrix)
    class_multilabels = ["action", "animation", "comedy", "crime", "drama", "musical", "romance", "thriller"]
    for i in range(len(multi_label_confusion_matrix)):
        matrix = multi_label_confusion_matrix[i]
        print(class_multilabels[i])
        TP = matrix[1][1]
        FP = matrix[0][1]
        FN = matrix[1][0]
        TN = matrix[0][0]
        print("Recall", TP / (TP + FN))
        print("Precision", TP / (TP + FP))
