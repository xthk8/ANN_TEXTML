import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 데이터 읽어오기
def read_reviews_from_folders(base_path, max_files_per_score=2000):
    data = []
    labels = []
    for score in range(1, 6):
        folder_path = os.path.join(base_path, str(score))
        file_count = 0
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

# 텍스트 토큰화 및 패딩
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(reviews)
sequences = tokenizer.texts_to_sequences(reviews)
maxlen = 25
data = pad_sequences(sequences, maxlen=maxlen)

# 원-핫 인코딩
encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
encoder.fit(np.array(scores).reshape(-1, 1))
labels = encoder.transform(np.array(scores).reshape(-1, 1)).toarray()

# 활성화 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

# 신경망 초기화
def initialize_network(input_size, hidden_size1, hidden_size2, output_size):
    np.random.seed(42)
    return {
        'W1': np.random.rand(input_size, hidden_size1),
        'b1': np.random.rand(hidden_size1),
        'W2': np.random.rand(hidden_size1, hidden_size2),
        'b2': np.random.rand(hidden_size2),
        'W3': np.random.rand(hidden_size2, output_size),
        'b3': np.random.rand(output_size)
    }

# 순전파 함수
def forward(network, x):
    W1, b1 = network['W1'], network['b1']
    W2, b2 = network['W2'], network['b2']
    W3, b3 = network['W3'], network['b3']
    
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

# 손실 함수
def cross_entropy_error(y, t):
    if t.ndim == 1:
        t = np.eye(y.shape[1])[t - 1]
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t.argmax(axis=1)] + 1e-7)) / batch_size

# 손실 함수를 이용한 손실 계산
def loss(network, x, t):
    y = forward(network, x)
    return cross_entropy_error(y, t)

# 기울기 계산
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val
        it.iternext()
    return grad

# 매개변수 갱신
def develop_gradient(network, x, t):
    loss_W = lambda W: loss(network, x, t)

    grads = {}
    for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
        grads[key] = numerical_gradient(loss_W, network[key])
    return grads

# K-fold cross-validation
def perform(network, data, labels, k=5, epochs=10, batch_size=100):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_accuracies = []

    for train_index, test_index in kf.split(data):
        train_data, test_data = data[train_index], data[test_index]
        train_labels, test_labels = labels[train_index], labels[test_index]

        # 매 폴드마다 네트워크 초기화
        network = initialize_network(input_size, hidden_size1, hidden_size2, output_size)
        
        for epoch in range(epochs):
            learning_rate = 0.01
            for i in range(0, len(train_data), batch_size):
                x_batch = train_data[i:i+batch_size]
                t_batch = train_labels[i:i+batch_size]

                grads = develop_gradient(network, x_batch, t_batch)
                for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                    network[key] -= learning_rate * grads[key]

        # 폴드별 정확도 계산
        test_accuracy = 0
        for i in range(0, len(test_data), batch_size):
            x_test_batch = test_data[i:i+batch_size]
            t_test_batch = test_labels[i:i+batch_size]

            y_test_batch = forward(network, x_test_batch)
            test_accuracy += np.sum(np.argmax(y_test_batch, axis=1) == np.argmax(t_test_batch, axis=1))
        test_accuracy /= len(test_data)
        fold_accuracies.append(test_accuracy)

    mean_accuracy = np.mean(fold_accuracies)
    return mean_accuracy

# 모델 초기화 및 K-fold cross-validation 실행
input_size = maxlen
hidden_size1 = 50
hidden_size2 = 5
output_size = 5

network = initialize_network(input_size, hidden_size1, hidden_size2, output_size)
mean_accuracy = perform(network, data, labels, k=5, epochs=10, batch_size=100)
print("Mean Accuracy :", mean_accuracy)

# Mean Accuracy : 0.2