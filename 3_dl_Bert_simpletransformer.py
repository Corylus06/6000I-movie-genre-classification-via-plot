# using simpletransformers to do the multi-label text classification

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
from simpletransformers.classification import MultiLabelClassificationModel
from sklearn.metrics import multilabel_confusion_matrix
# torch_version, transformers_version, tokenizers_version
# 1.6.0+cu101, 4.0.0, 0.9.3


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

def create_multilabel_model():
    model_args = {
        "reprocess_input_data": True,
        "overwrite_output_dir": True,
        "save_model_every_epoch": False,
        "save_eval_checkpoints": False,
        "max_seq_length": 512,
        "train_batch_size": 4,
        "num_train_epochs": 1,
        "threshold": 0.5,
    }

    model = MultiLabelClassificationModel('bert', 'bert-base-uncased', num_labels=8, args=model_args)

    return model

# multi-label
if __name__ == "__main__":
    movies_data = load_one_hot_data()
    genre_nums = 8
    print(movies_data)
    train_data, test_data = train_test_split(movies_data, test_size=0.1)
    test_label = test_data["label"].tolist()
    print(test_label)
    print(test_data)
    print(test_data["Plot"].tolist()[0])
    print(test_data["Plot"].tolist()[1])
    model = create_multilabel_model()
    model.train_model(train_data[["Plot", "label"]])
    # result, model_outputs, wrong_predictions = model.eval_model(test_data[["Plot", "label"]])
    preds, outputs = model.predict(test_data["Plot"].tolist())
    test_label = test_data["label"].tolist()
    print(preds)
    # print(outputs)
    # print(result)

    multi_label_confusion_matrix = multilabel_confusion_matrix(test_label, preds)
    print(multi_label_confusion_matrix)
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