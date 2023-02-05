# -*- coding: utf-8 -*-

import os
import sys
import glob
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch


def make_dataset_list(meta_csv_path, header=1, phase='train'):
    """[meta_csvファイルを読み込んでファイルパスのリストを返す]
    
    Arguments:
        meta_csv {[type]} -- [description]
    
    Keyword Arguments:
        header {[type]} -- [description] (default: {True})
    """

    # csvファイルの読み込み
    meta_csv_df = pd.read_csv(meta_csv_path, sep=',', header=header)

    # 各行のデータをリストに入れる
    index_list = []
    path_list = []
    label_list = []
    for record in meta_csv_df.itertuples(name=None):
        # 行番号
        index_list.append(record[0]) 

        # OSの種類によってファイルパスの区切り文字を揃える
        path = None
        if os.name == 'nt': 
            # Windowsの場合
            path = re.sub(r"/+", r"\\", record[1])
        else :
            # Unix系の場合
            path = re.sub(r"\\+", r"/", record[1])

        # ファイルパスを取得
        path_list.append(path)

        # ラベル
        if phase == 'train' or phase == 'val':
            label_list.append(record[2])
    
    # meta_csvファイルがあるフォルダ名をpath_listに追加
    meta_csv_dirname = os.path.dirname(meta_csv_path)
    for i, path in enumerate(path_list):
        path_list[i] = meta_csv_dirname + os.sep + str(path_list[i])

    data_dict = { "index": index_list, "path": path_list, "label": label_list }
    return data_dict


class LSTMDataset(Dataset):
    """[torchベースのLSTMデータセット]
    
    Arguments:
        Dataset {[type]} -- [description]
    """

    def __init__(self, file_list, label_list, normalize=None, phase="train"):
        self.file_list = file_list
        self.label_list = []
        self.phase = phase
        self.normalize = normalize
        self.target_dim = np.array(label_list).ndim

        # Not Compatible One-Hot-Vector
        if self.target_dim > 1: 
            for index in np.argmax(np.array(label_list), axis=1):
                self.label_list.append(index)
        else:
            self.label_list = label_list


    def __len__(self):
        """[torchのDataloaderに必要な機能その１]
        
        Returns:
            [type] -- [データセットの数を返す]
        """
        return len(self.file_list)

    def __getitem__(self, index):
        """[torchのDataloaderに必要な機能その2]
        
        Arguments:
            index {[type]} -- [indexで指定したファイルを1つ取得]
        """
        csv_file_df = pd.read_csv(self.file_list[index], sep=',', header=None)
        data_list = csv_file_df.iloc[:, 0].values.tolist()

        if self.normalize:
            # 値を0~1に規格化
            data_max = max(data_list)
            data_min = min(data_list)
            data_list = list(map(lambda x: (x - data_min) / (data_max - data_min), data_list))
           
        label = self.label_list[index]
        
        return torch.tensor(data_list), torch.tensor(label)


def torchDataSet(meta_csv_train = None, meta_csv_val = None, normalize=None):

    # ファイルパスのリスト
    train_dict = make_dataset_list(meta_csv_train, phase='train')
    val_dict = make_dataset_list(meta_csv_val,  phase='val')
    print("train_num: {}, val_num: {}".format(len(train_dict["index"]), len(val_dict['index'])))

    # torch用DataSet
    train_dataset = LSTMDataset(file_list=train_dict['path'],
                                label_list=train_dict["label"], 
                                normalize=normalize, 
                                phase='train')

    val_dataset = LSTMDataset(file_list=val_dict['path'],
                              label_list=val_dict['label'],
                              normalize=normalize,
                              phase='val')

    print("train_dataset: {}, val_dataset: {}".format(type(train_dataset), type(val_dataset)))

    dataset_dict = { 'train': train_dataset, 'val': val_dataset }

    return dataset_dict


if __name__ == "__main__":

    meta_csv_train = "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/train_meta-csv.csv"
    meta_csv_val   = "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/validation_meta-csv.csv"
    meta_csv_test  = "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/test_meta-csv.csv"

    train_dict = make_dataset_list(meta_csv_train, phase='train')
    val_dict = make_dataset_list(meta_csv_val,  phase='val')
    test_dict = make_dataset_list(meta_csv_test, phase='test')
    #print(train_dict)
    #print(train_dict['index'][:5])
    #print(train_dict['path'][:5])
    #print(train_dict['label'][:5])
    print("train_num: {}, val_num: {}, test_num: {}".format(len(train_dict["index"]), len(val_dict['index']), len(test_dict['index'])))

    train_dataset = LSTMDataset(file_list=train_dict['path'],label_list=train_dict["label"], normalize=None, phase='train')
    val_dataset = LSTMDataset(file_list=val_dict['path'], label_list=val_dict['label'], normalize=None, phase='val')
    #print("train_dataset: {}".format(type(train_dataset)))
    #print("val_dataset: {}".format(type(val_dataset)))

    # データの確認
    print("train_dict['path'][0]: ", train_dict['path'][0])
    csv_file_df = pd.read_csv(train_dict['path'][0], sep=',', header=None)
    data = csv_file_df.iloc[:, 0].values.tolist()
    print("type(data): ", type(data))
    print("data_len: ", len(data))
    print("data: ", data)
    data_max = max(data)
    data_min = min(data)
    data = list(map(lambda x: (x - data_min) / (data_max - data_min), data))
    print("normalized data: ", data)


    _, target = train_dataset[0]
    print("target: ", target)
    
    


    