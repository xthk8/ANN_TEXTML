import os
import glob
import numpy as np
import random
from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

###################################

class Relu:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx


class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = np.exp(-np.abs(x))
        return np.where(x > 0, 1 / (1 + out), out / (1 + out))

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        return dx


def softmax(x):
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        dx = (self.y - self.t) / batch_size
        return dx


#####################################################################
    
# 전체 신경망 구현 
    
class TwoLayerNet:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std=0.01):
        self.params = {
            'W1': weight_init_std * np.random.randn(input_size, hidden_size1),
            'b1': np.zeros(hidden_size1),
            'W2': weight_init_std * np.random.randn(hidden_size1, hidden_size2),
            'b2': np.zeros(hidden_size2),
            'W3': weight_init_std * np.random.randn(hidden_size2, output_size),
            'b3': np.zeros(output_size)
        }

        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        self.lastLayer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1:
            t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        grad = {}
        grad['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grad['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grad['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grad['b2'] = numerical_gradient(loss_W, self.params['b2'])
        grad['W3'] = numerical_gradient(loss_W, self.params['W3'])
        grad['b3'] = numerical_gradient(loss_W, self.params['b3'])
        return grads

    def gradient(self, x, t):

        # 순전파
        self.loss(x, t)

        # 역전파
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db
        grads['W3'] = self.layers['Affine3'].dW
        grads['b3'] = self.layers['Affine3'].db        

        return grads
    
    #########################################################
    ## 옵티마이저

class SGD:

    """확률적 경사 하강법（Stochastic Gradient Descent）"""

    def __init__(self, lr):
        self.lr = lr
        
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key] 


class Momentum:

    """모멘텀 SGD"""

    def __init__(self, lr, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None
        
    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():                                
                self.v[key] = np.zeros_like(val)
                
        for key in params.keys():
            self.v[key] = self.momentum*self.v[key] - self.lr*grads[key] 
            params[key] += self.v[key]


class AdaGrad:

    """AdaGrad"""

    def __init__(self, lr):
        self.lr = lr
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class RMSprop:

    """RMSprop"""

    def __init__(self, lr, decay_rate = 0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None
        
    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
            
        for key in params.keys():
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)


class Adam:

    """Adam (http://arxiv.org/abs/1412.6980v8)"""

    def __init__(self, lr, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)
            

###################################

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


def initialize_network(input_size, hidden_size1, hidden_size2, output_size):
    np.random.seed(42)
    return {
        'W1': np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size),    # ReLU 이용 -> He 초깃값
        'b1': np.zeros(hidden_size1),
        'W2': np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1),
        'b2': np.zeros(hidden_size2),
        'W3': np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2),
        'b3': np.zeros(output_size)
    }

############################################################################


# 맨 처음 - 데이터 읽어오기
def read_reviews_from_folders(base_path, max_files_per_score=2000):
    data = []
    labels = []
    for score in range(1, 6):
        folder_path = os.path.join(base_path, str(score))
        file_paths = glob.glob(os.path.join(folder_path, '*.txt'))
        random.shuffle(file_paths)  # 데이터 무작위로 섞기
        file_count = 0

        for file_path in file_paths:
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
train_reviews, test_reviews, train_scores, test_scores = train_test_split(
    reviews, scores, test_size=0.2, random_state=42
)

#########################################################################################

# 텍스트 토큰화 및 패딩
tokenizer = Tokenizer(num_words=25)  # 입력층 뉴런 수와 동일화 (단어 단위로 입력층에 들어감) 
tokenizer.fit_on_texts(reviews)  # 전체 리뷰 데이터에 대해 fit
train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

# 모든 시퀀스를 같은 길이로 맞춤
maxlen = 25
train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

# 원-핫 인코딩 적용
encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
encoder.fit(np.array(train_scores).reshape(-1, 1))  # 훈련 점수에 대해서만 fit
train_labels = encoder.transform(np.array(train_scores).reshape(-1, 1)).toarray()
test_labels = encoder.transform(np.array(test_scores).reshape(-1, 1)).toarray()

# 신경망 구조 설정 및 초기화
input_size = maxlen 
hidden_size1 = 50
hidden_size2 = 5
output_size = 5

#################################################

# 실행 함수 정의
def perform(train_data, train_labels, test_data, test_labels, epochs, batch_size):

    accuracies = []
    params = initialize_network(input_size, hidden_size1, hidden_size2, output_size)
    network = TwoLayerNet(input_size, hidden_size1, hidden_size2, output_size)
    optimizer = Adam(lr=0.002)
    # learning_rate = 0.1

    for epoch in range(epochs):

        test_accuracy = 0

        for i in range(0, len(train_data), batch_size):

            # 학습 (훈련)
            x_batch = train_data[i:i+batch_size]
            t_batch = train_labels[i:i+batch_size]

            t_batch = np.array(t_batch)

            grad = network.gradient(x_batch, t_batch)
            params = network.params
            optimizer.update(params, grad)

            #for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
            #    params[key] -= learning_rate * grad[key]


            # 테스트 (평가)
            x_test_batch = test_data[i:i+batch_size]
            t_test_batch = test_labels[i:i+batch_size]

            y_test_batch = network.predict(x_test_batch)
            test_accuracy += np.sum(np.argmax(y_test_batch, axis=1) == np.argmax(t_test_batch, axis=1))

        test_accuracy /= len(test_data)
        accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy}")

    mean_accuracy = np.mean(accuracies)
    return accuracies, mean_accuracy


acclst, mean_accuracy = perform(train_data, train_labels, test_data, test_labels, epochs=10, batch_size=100)
seq = [i for i in range(1,11)]

plt.plot(seq, acclst)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()
