# python utils
import os.path as ospath
import random
from tqdm import tqdm
import numpy as np
import numpy.random as np_random
import matplotlib.pyplot as plt

# pytorch 
import torch
import torch.random as torch_random
import torch.jit
import torch.utils.data as torch_data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#import torch.utils.tensorboard
# import torch.nn.LSTM as LSTM
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False

# 共通ランダムシード
random.seed(1234)
np_random.seed(1234)
torch_random.manual_seed(1234)


class LSTMClassifier(nn.Module):

    def __init__(self, input_dim, hidden_dim, class_dim=1, num_layers=1):
        super(LSTMClassifier, self).__init__()

        self.input_dim = input_dim   # 1データあたりの次元
        self.hidden_dim = hidden_dim # 隠れ層の次元
        self.class_dim = class_dim   # 出力(確信度)の次元
        self.num_layers = num_layers # 出力方向に存在するLSTMセルの数

        """
        本来LSTMの入力は先述したように、Sequence_Length x Batch_Size x Vector_Sizeの形式になっていなければなりません。
        しかし、batch_first=Trueとすると、Batch_Size x Sequence_Length x Vector_Sizeの形式で入力することができるようになります。
        この辺りは結構ややこしく、次元を合わせないと動かないので注意しましょう。
        """
        self.lstm = nn.LSTM(self.input_dim, 
                            self.hidden_dim, 
                            batch_first = True, 
                            num_layers = self.num_layers) # LSTMモジュール

        # LSTMの出力からsoftmaxの手前までの値を出力するAffine変換
        self.outputLayer = nn.Linear(self.hidden_dim, self.class_dim) 

        # LogSoftmax
        #self.lastActivation = nn.LogSoftmax(dim = 1) # 行方向で確率変換
        #self.lastActivation = torch.sigmoid


    def forward(self, x, hidden0=None):
        # xは[batch_size, input_dim]
        output, (last_hideen, last_cell) = self.lstm(x, hidden0)
        """
        欲しいのは、outputの一番最後の値だけなので、output[:, -1, :]で時系列の最後の値（ここではベクトル）を取り出します。
        """
        output = output[:, -1, :] # many-to-oneモデルを採用するので、時系列方向に展開したLSTMモジュールの最後の出力のみを使う. ちなみにlast_hiddenと同じになる
        score = self.outputLayer(output)
        return score
        #logProb = self.logsoftmax(score)
        #return logProb
        #return self.lastActivation(score)


def save_cpu_model_and_weights(filename, model, args=[], kwargs={}):
    import inspect
    import pickle
    from copy import deepcopy

    """モデルと重みの保存"""
    cpu_model = deepcopy(model).cpu()

    state = {
        "module_path": inspect.getmodule(model, _filename=True).__file__, # モデルのクラスが定義されているファイルのパス
        "class_name": model.__class__.__name__, # モデルのクラス名
        "state_dict": cpu_model.state_dict(),   # モデルの重み
        "args": args,    # モデルの引数（リスト引数）
        "kargs": kwargs, # モデルの引数（キーワード引数）
    }

    with open(filename, "wb") as f:
        pickle.dump(state, f)


def load_cpu_model_and_weights(filename):
    import pickle
    from importlib import machinery

    """モデルと重みの読み込み"""
    with open(filename, "rb") as f:
        state = pickle.load(f)

    module = machinery.SourceFileLoader(state["module_path"], state["module_path"]).load_module()
    args, kwargs = state["args"], state["kargs"]                     # モデルパラメータの読み込み
    cpu_model = getattr(module, state["class_name"])(*args, **kwargs) # モデルクラスの読み込み
    cpu_model.load_state_dict(state["state_dict"])                   # モデル重みの読み込み
    """
    ＜注意＞
        save_model()で保存されるファイルパスは今いるPC上での絶対パスなので，保存ファイルを別PCに転送して使う場合には，
        load_model()で読み込めない可能性があります。（絶対パスが同じ表記なら問題なし）
        load_model()が使えない場合は，保存ファイルからstate_dict, args, kwargsだけ取り出して使用して下さい。
    """
    return cpu_model




def predict_test():

    # 入力データの情報
    input_dim = 1
    hidden_dim = 10
    class_dim = 1
    timeSeries = 25
    data_num = 3

    # 入力データを作成
    input_list = []
    for i in range(data_num):
        input = []
        for data in range(timeSeries):
            data += random.uniform(-2.0, +2.0)
            input.append(data)
        input_list.append(input)

    input_numpy = np.array(input_list)

    #x = np.array([ [ i for i in range(input_dim)] for k in range(data_num) ])
    print("Input -> Size({0},{1}), Data: {2}".format(input_numpy.shape[0], input_numpy.shape[1], input_numpy))
    #for k in range(data_num):
    #    plt.plot(range(input_dim), input_numpy[k])
    #plt.show()

    # テスト
    lstmModel = LSTMClassifier(input_dim=input_dim, hidden_dim=hidden_dim, class_dim=class_dim)

    # 誤差
    criterion = nn.CrossEntropyLoss()
    
    # 最適化手法
    optimizer = optim.Adam(lstmModel.parameters(), lr=0.01)

    # モデルのパラメータを初期化
    optimizer.zero_grad()

    input_tensor = torch.from_numpy(input_numpy)
    input_tensor = torch.unsqueeze(input_tensor, dim=2)
    input_tensor = input_tensor.float() # pytorchのTensorはfloat型にする
    print("Tensor -> Size:{} Data: {}".format(input_tensor.size(), input_tensor))

    # 順伝播
    output = lstmModel(input_tensor)
    print(output)
    #print("output -> Size:{}, Data:{}".format(output.numpy))


if __name__ == "__main__":

    """パラメータ"""
    params = {
        "meta_csv_train": "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/train_meta-csv.csv",
        "meta_csv_val": "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/validation_meta-csv.csv",
        "meta_csv_test": "/Users/inoueshinichi/Desktop/01_ML/NeuralNetwork/LSTM/dummyDataset/dataset/test_meta-csv.csv", 
        "lstm_input_dim": 1,
        "lstm_hidden_dim": 10,
        "lstm_output_dim": 1,
        "lstm_time_series": 25,
        "adam_lr": 0.01,
        "epochs": 1,
        "batch_size": 30,
        "class_labels": [0, 1],
        "normalize": True,
    }
    save_model_and_weights_name = "lstm_model_inDim{0}_hiddenDim{1}_outputDim{2}.pkl".format(params['lstm_input_dim'], 
                                                                                 params['lstm_hidden_dim'], 
                                                                                 params['lstm_output_dim'])
    save_model_and_weights_path = "./trained_model/{0}".format(save_model_and_weights_name)

    """データセット"""
    from LSTM_Dataset import torchDataSet
    dataset_dict = torchDataSet(meta_csv_train=params['meta_csv_train'],
                                meta_csv_val=params['meta_csv_val'],
                                normalize=params['normalize'])
    
    """データローダー"""
    from LSTM_DataLoader import torchDataLoader
    dataloader_dict = torchDataLoader(batch_size=params['batch_size'], 
                                      train_dataset=dataset_dict['train'],
                                      val_dataset=dataset_dict['val'])


    """CPU or GPU 上のメモリで計算する"""
    device_type = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス: ", device_type)

    if torch.cuda.is_available():
        print("デバイス名: ", torch.cuda.get_device_name(device_type))
        print("利用可能GPU数: ", torch.cuda.device_count())
        print("デバイス番号: ", torch.cuda.current_device())
        torch.backends.cudnn.deterministic = True # GPU計算が決定論的 or NOT. benchmark=Falseにすること
        torch.backends.cudnn.benchmark = True     # ネットワークの準伝播及び逆伝搬関数の計算手法が程度固定であれば、高速化される
        

    """LSTMネットワークモデル"""
    lstm_net = LSTMClassifier(input_dim=params['lstm_input_dim'], 
                              hidden_dim=params['lstm_hidden_dim'], 
                              class_dim=params['lstm_output_dim'])
    # 複数GPUが使用できる場合
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            print("{}GPUs. 複数GPUを用いた並列計算の設定を行います.".format(torch.cuda.device_count()))
            lstm_net = nn.DataParallel(lstm_net)
    # 設定されたデバイス上のメモリにネットワークオブジェクトを展開
    lstm_net.to(device_type)
    
    """誤差関数"""
    #criterion = nn.NLLLoss() # negative log likelihood loss (= clossentropyloss - logsoftmax)
    #criterion = nn.CrossEntropyLoss() # クラスラベルが(0 or 1 or 2 or ..)のとき
    #criterion = nn.BCELoss() # Binary Cross Entropy (0 or 1)のとき
    criterion = nn.BCEWithLogitsLoss() # sigmoidを含んだ２分類誤差関数のlog版. モデルのforwardの出力はscoreにすること

    """最適化手法"""
    optimizer = optim.Adam(lstm_net.parameters(), lr=params['adam_lr'])

    """学習"""
    loss_dict = {'train': [], 'val': [], 'all': []}
    correct_dict = {'train': [], 'val': [], 'all': []}
    accuracy_dict = {'train': [], 'val': [], 'all': []}
    for epoch in range(params['epochs']):
        print('Epoch {}/{}'.format(epoch + 1, params['epochs']))
        print('--------')

        for phase in ['train', 'val']:
            if phase == 'train':
                lstm_net.train() # 訓練モードのモデル
            else:
                lstm_net.eval()  # 検証モードのモデル

            epoch_loss = 0.0
            epoch_correct = 0
            epoch_accuracy = 0.0

            # 未学習時の検証性能を確かめるため、epoch=0の訓練では未学習のモデルを使用する
            #if (epoch == 0) and (phase == 'train'):
            #    continue

            if phase == "train":
                print("##### train #####")
            else:
                print("##### val ######")

            # データローダからミニバッチを取り出すループ
            for inputs, labels in tqdm(dataloader_dict[phase]):

                # LSTM対応 入力データは3次元, ラベルデータは2次元にする
                lstm_inputs = inputs.clone()
                lstm_inputs = lstm_inputs.unsqueeze(2)
                lstm_labels = labels.clone()
                lstm_labels = lstm_labels.unsqueeze(1).float()
                #print(inputs)
                #print(labels)
                #print("inputs type: {}, size: {}, data: {}".format(type(inputs), inputs.size(), inputs))
                #print("labels type: {}, size: {}, data: {}".format(type(labels), labels.size(), labels))

                # GPUが利用できるならGPUのメモリにデータを転送する
                lstm_inputs = lstm_inputs.to(device_type)
                lstm_labels = lstm_labels.to(device_type)
                
                # 各パラメータの勾配の初期化
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    # 順伝播
                    outputs = lstm_net(lstm_inputs)        # 出力
                    loss = criterion(outputs, lstm_labels) # 誤差

                    # 予測ラベル
                    preds = torch.where(torch.sigmoid(outputs.squeeze(1)) > 0.5, # 条件
                                        torch.tensor(params['class_labels'][1]), # True
                                        torch.tensor(params['class_labels'][0])) # False

                    # 訓練時は逆誤差伝播
                    if phase == 'train':    
                        loss.backward()  # 各パラメータの勾配を求める
                        optimizer.step() # 算出した勾配から各パラメータを更新 
                    

                    # epoch_lossの更新
                    epoch_loss += loss.item() * inputs.size(0) # 1データに対する平均loss * バッチサイズ = バッチサイズにおけるloss

                    # epoch_correctの更新
                    epoch_correct += torch.sum(labels == preds)

            # epoch_accuracyの更新
            epoch_accuracy = epoch_correct.double() / len(dataloader_dict[phase])

            # 訓練結果 or 検証結果    
            loss_dict[phase].append(epoch_loss)
            correct_dict[phase].append(epoch_correct)
            accuracy_dict[phase].append(epoch_accuracy)

        # データセット全体の結果
        loss_dict['all'].append(loss_dict['train'][-1] + loss_dict['val'][-1])
        correct_dict['all'].append(correct_dict['train'][-1] + correct_dict['val'][-1])
        accuracy_dict['all'].append(correct_dict['all'][-1] / (len(dataloader_dict['train']) + len(dataloader_dict['val'])))

        # epoch毎の結果の表示
        print("LOSS     -> train: {0:.4f}, val: {1:.4f}, all: {2:.4f}".format(loss_dict['train'][-1], 
                                                                              loss_dict['val'][-1],
                                                                              loss_dict['all'][-1]))
        print("CORRECT  -> train: {0}, val: {1}, all: {2}".format(correct_dict['train'][-1], 
                                                                  correct_dict['val'][-1], 
                                                                  correct_dict['all'][-1]))
        print("ACCURACY -> train: {0:.4f},  val: {1:.4f},  all: {2:.4f}".format(accuracy_dict['train'][-1], 
                                                                                accuracy_dict['val'][-1], 
                                                                                accuracy_dict['all'][-1]))

    """モデルと重みの保存"""
    print("モデルと重みを保存します。保存先: ", save_model_and_weights_path)
    print("保存するモデル: ", lstm_net)
    save_cpu_model_and_weights(filename=save_model_and_weights_path,
                               model=lstm_net,
                               args=[params['lstm_input_dim'], 
                                     params['lstm_hidden_dim'], 
                                     params['lstm_output_dim']]
                               )

    """モデルと重みの読み出し"""
    print("モデルと重みを読み出します。読み出し先: ", save_model_and_weights_path)
    loaded_net = load_cpu_model_and_weights(save_model_and_weights_path)
    print("読み出したモデル: ", loaded_net)


    import matplotlib.pyplot as plt

    """誤差と精度のグラフ表示"""
    print("loss_dict[train] :", loss_dict["train"])
    # loss
    #for phase in ['train', 'val', 'all']:
    #    plt.plot(loss_dict[phase], label=phase)
    #
    #plt.legend()
    #plt.show()

    """"テストデータによる予測精度の確認"""
    # テストデータ
    import pandas as pd
    import re
    import os
    meta_csv_df = pd.read_csv(params['meta_csv_test'], sep=',', header=1)
    test_path_list = []
    for record in meta_csv_df.itertuples(name=None):    
        # OSの種類によってファイルパスの区切り文字を揃える
        path = None
        if os.name == 'nt': 
            # Windowsの場合
            path = re.sub(r"/+", r"\\", record[1])
        else :
            # Unix系の場合
            path = re.sub(r"\\+", r"/", record[1])
        # ファイルパスを取得
        test_path_list.append(path)

    # meta_csvファイルがあるフォルダ名をpath_listに追加
    meta_csv_dirname = os.path.dirname(params['meta_csv_test'])
    for i, path in enumerate(test_path_list):
        test_path_list[i] = meta_csv_dirname + os.sep + str(test_path_list[i])

    # 勾配の自動計算をOFF
    with torch.no_grad():
        for file_path in test_path_list:

            # Rawデータを抽出
            csv_file_df = pd.read_csv(file_path, sep=',', header=None)
            data_list = csv_file_df.iloc[:, 0].values.tolist()

            if params['normalize']:
                # 値を0~1に規格化
                data_max = max(data_list)
                data_min = min(data_list)
                data_list = list(map(lambda x: (x - data_min) / (data_max - data_min), data_list))

                # list -> tensor
                lstm_inputs = torch.tensor(data_list, dtype=torch.float32)
                lstm_inputs = lstm_inputs.unsqueeze(1).unsqueeze(0)

                # 予測
                test_outputs = lstm_net(lstm_inputs) # 出力
                test_preds = torch.where(torch.sigmoid(test_outputs.squeeze(1)) > 0.5, # 条件
                                         torch.tensor(params['class_labels'][1]), # True
                                         torch.tensor(params['class_labels'][0])) # False

            print("Predicted Label: {}".format(test_preds))
        

    print("###### Finish ######")

    

    


