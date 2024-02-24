import os, sys
sys.path.append(os.pardir) 
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from common.multi_layer_net_extend import MultiLayerNetExtend
from common.optimizer import SGD, Adam

###################################

#####################################################################
    
# 전체 신경망 구현 
    
class NN:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size1),
            'b1': np.zeros(hidden_size1),
            'gamma1': np.ones(hidden_size1),   # 추가: 배치 정규화 파라미터
            'beta1': np.zeros(hidden_size1),   # 추가: 배치 정규화 파라미터
            'W2': weight_init_std * np.random.randn(hidden_size1, hidden_size2),
            'b2': np.zeros(hidden_size2),
            'gamma2': np.ones(hidden_size2),   # 추가: 배치 정규화 파라미터
            'beta2': np.zeros(hidden_size2),   # 추가: 배치 정규화 파라미터
            'W3': weight_init_std * np.random.randn(hidden_size2, output_size),
            'b3': np.zeros(output_size),
            'gamma3': np.ones(output_size),    # 추가: 배치 정규화 파라미터
            'beta3': np.zeros(output_size)     # 추가: 배치 정규화 파라미터
        }

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['BatchNorm1'] = batch_norm(self.params['gamma1'], self.params['beta1'])  # 추가: 배치 정규화 레이어
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['BatchNorm2'] = batch_norm(self.params['gamma2'], self.params['beta2'])  # 추가: 배치 정규화 레이어
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.layers['BatchNorm3'] = batch_norm(self.params['gamma3'], self.params['beta3'])  # 추가: 배치 정규화 레이어
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x, train_flg=True):

        x = np.array(x)

        for layer in self.layers.values():
            if isinstance(layer, batch_norm):
                x = layer.forward(x, train_flg)
            else:
                x = layer.forward(x)
        return x

    def loss(self, x, t, train_flg=True):

        x = np.array(x)
        
        y = self.predict(x, train_flg)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t, train_flg=False):
        y = self.predict(x, train_flg)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(len(x))
        return accuracy

    def gradient(self, x, t):
        self.loss(x, t)

        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {
            'W1': self.layers['Affine1'].dW,
            'b1': self.layers['Affine1'].db,
            'gamma1': self.layers['BatchNorm1'].dgamma,  # 추가: 배치 정규화 파라미터의 기울기
            'beta1': self.layers['BatchNorm1'].dbeta,    # 추가: 배치 정규화 파라미터의 기울기
            'W2': self.layers['Affine2'].dW,
            'b2': self.layers['Affine2'].db,
            'gamma2': self.layers['BatchNorm2'].dgamma,  # 추가: 배치 정규화 파라미터의 기울기
            'beta2': self.layers['BatchNorm2'].dbeta,    # 추가: 배치 정규화 파라미터의 기울기
            'W3': self.layers['Affine3'].dW,
            'b3': self.layers['Affine3'].db,
            'gamma3': self.layers['BatchNorm3'].dgamma,  # 추가: 배치 정규화 파라미터의 기울기
            'beta3': self.layers['BatchNorm3'].dbeta     # 추가: 배치 정규화 파라미터의 기울기
        }

        return grads



###################################

############################################################################


# 맨 처음 - 데이터 읽어오기
def read_reviews_from_folders(base_path, max_files_per_score=2000):
    data = []
    labels = []
    for score in range(1, 6):
        folder_path = os.path.join(base_path, str(score))
        file_count = 0
        # 각각의 점수에 해당하는 텍스트 파일 읽기 (최대 2000개)
        for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
            if file_count >= max_files_per_score:
                break
            with open(file_path, 'r', encoding='utf8') as file:
                review_text = file.read().strip()
                data.append(review_text)
                labels.append(score)
            file_count += 1
    return data, labels

# 파일 경로 정의 및 함수 호출
review_base_path = "C:/Users/USER/Desktop/학부연구/밑바닥부터 시작하는 딥러닝/reviews3"
reviews, scores = read_reviews_from_folders(review_base_path)

# 데이터셋 분할
x_train, t_train, x_test, t_test = train_test_split(
    reviews, scores, test_size=0.2, random_state=42
)

#########################################################################################

# 텍스트 토큰화 및 패딩
tokenizer = Tokenizer(num_words=25)  # 입력층 뉴런 수와 동일화 (단어 단위로 입력층에 들어감) 
tokenizer.fit_on_texts(reviews)  # 전체 리뷰 데이터에 대해 fit
train_sequences = tokenizer.texts_to_sequences(x_train)
test_sequences = tokenizer.texts_to_sequences(t_train)

# 모든 시퀀스를 같은 길이로 맞춤
maxlen = 25
train_data = pad_sequences(train_sequences, maxlen=maxlen)
print(train_data)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

# 원-핫 인코딩 적용
encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
encoder.fit(np.array(x_test).reshape(-1, 1))  # 훈련 점수에 대해서만 fit
train_labels = encoder.transform(np.array(x_test).reshape(-1, 1)).toarray()
test_labels = encoder.transform(np.array(t_test).reshape(-1, 1)).toarray()

# 신경망 구조 설정 및 초기화
input_size = maxlen 
hidden_size1 = 50
hidden_size2 = 5
output_size = 5

#################################################

# 배치정규화


def batch_norm(weight_init_std):

    bn_network = MultiLayerNetExtend(input_size=25,
                                     hidden_size_list=[50, 5],   
                                     output_size=5,
                                     weight_init_std=weight_init_std,
                                     use_batchnorm=True)
    network = MultiLayerNetExtend(input_size=25,
                                  hidden_size_list=[50, 5],
                                  output_size=5,
                                  weight_init_std=weight_init_std)
    optimizer = SGD(lr=0.1)

    train_acc_list = []
    bn_train_acc_list = []
    train_size = train_data.shape[0]

    iter_per_epoch = max(train_size / 100, 1)
    epoch_cnt = 20

    ### 옵티마이저
    for i in range(epoch_cnt):

        x_batch = x_train[i:i+100]
        t_batch = t_train[i:i+100]

        for _network in (bn_network, network):
            grads = _network.gradient(x_batch, t_batch)
            optimizer.update(_network.params, grads)

        if i % iter_per_epoch == 0:
            train_acc = network.accuracy(x_train, t_train)
            bn_train_acc = bn_network.accuracy(x_train, t_train)
            train_acc_list.append(train_acc)
            bn_train_acc_list.append(bn_train_acc)

            print("epoch:" + str(epoch_cnt) + " | " + str(train_acc) + " - "
                  + str(bn_train_acc))

    return train_acc_list, bn_train_acc_list



# 그래프 그리기==========
weight_scale_list = np.logspace(0, -4, num=16)
x = np.arange(25)

for i, w in enumerate(weight_scale_list):
    print("============== " + str(i+1) + "/16" + " ==============")
    train_acc_list, bn_train_acc_list = batch_norm(w)

    plt.subplot(4, 4, i+1)
    plt.title("W:" + str(w))
    if i == 15:
        plt.plot(x, bn_train_acc_list,
                 label='Batch Normalization', markevery=2)
        plt.plot(x, train_acc_list, linestyle="--",
                 label='Normal(without BatchNorm)', markevery=2)
    else:
        plt.plot(x, bn_train_acc_list, markevery=2)
        plt.plot(x, train_acc_list, linestyle="--", markevery=2)

    plt.ylim(0, 1.0)
    if i % 4:
        plt.yticks([])
    else:
        plt.ylabel("accuracy")
    if i < 12:
        plt.xticks([])
    else:
        plt.xlabel("epochs")
    plt.legend(loc='lower right')

plt.show()

##############################################################################################

'''
# 실행 함수 정의
def perform(params, train_data, train_labels, test_data, test_labels, epochs=20, batch_size=100):
    accuracies = []

    for epoch in range(epochs):
        learning_rate = 0.1

        for i in range(0, len(train_data), 100):
            x_batch = train_data[i:i+100]
            t_batch = train_labels[i:i+100]

            t_batch = np.array(t_batch)

            grads = network.gradient(x_batch, t_batch)

            for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                params[key] -= learning_rate * grads[key]

        test_accuracy = 0
        for i in range(0, len(test_data), 100):
            x_test_batch = test_data[i:i+100]
            t_test_batch = test_labels[i:i+100]

            y_test_batch = network.predict(x_test_batch)
            test_accuracy += np.sum(np.argmax(y_test_batch, axis=1) == np.argmax(t_test_batch, axis=1))

        test_accuracy /= len(test_data)
        accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy}")

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy


params = initialize_network(input_size, hidden_size1, hidden_size2, output_size)
network = NN(input_size, hidden_size1, hidden_size2, output_size)
mean_accuracy = perform(params, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=100)
print("Mean Accuracy:", mean_accuracy)
'''


