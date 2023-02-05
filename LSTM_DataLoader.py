import os
import sys
import numpy as np
import torch


def torchDataLoader(batch_size = 5, train_dataset = None, val_dataset = None):
    
    """ DataLoader作成 """

    # train
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size = batch_size, shuffle = True
    )

    # val
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset, batch_size = batch_size, shuffle = True
    )

    # 辞書にまとめる
    datasetloader_dict = { 'train': train_dataloader, 'val': val_dataloader }

    # # 動作確認
    # batch_iterator = iter(datasetloader_dict['train']) # イテレータに変換
    # inputs, labels = next(batch_iterator) # 1番目の要素を取り出す
    # print(inputs.size())
    # print(labels)

    # print("train_dataloader: {}, val_dataloader: {}".format(type(train_dataloader), type(val_dataloader)))

    return datasetloader_dict



if __name__ == "__main__":
    
    params = {
        "meta_csv_train": "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/train_meta-csv.csv",
        "meta_csv_val": "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/validation_meta-csv.csv",
        "meta_csv_test": "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/test_meta-csv.csv",
    }

    from LSTM_Dataset import torchDataSet
    torch_datast_dict = torchDataSet(meta_csv_train = params["meta_csv_train"], 
                                     meta_csv_val = params['meta_csv_val'])
        
    torch_dataloader_dict = torchDataLoader(batch_size = 5, 
                                            train_dataset = torch_datast_dict['train'],
                                            val_dataset = torch_datast_dict['val'])

    # 動作確認
    batch_iterator = iter(torch_dataloader_dict['train']) # イテレータに変換
    inputs, labels = next(batch_iterator) # 1番目の要素を取り出す

    inputs = inputs.unsqueeze(2)
    labels = labels.unsqueeze(1)
    
    print("inputs -> type:{}, data: {}".format(type(inputs), inputs))
    print("labels -> type:{}, data: {}".format(type(labels), labels))
    print("input_size: {}, {}, {} dim: {}".format(inputs.size()[0], inputs.size()[1], inputs.size()[2],inputs.ndim))

