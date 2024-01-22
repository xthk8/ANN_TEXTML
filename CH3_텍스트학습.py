import os
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

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
review_base_path = "C:/Users/USER/Desktop/학부연구/밑바닥부터 시작하는 딥러닝/reviews_2"
reviews, scores = read_reviews_from_folders(review_base_path)


# 데이터셋 분할
train_reviews, test_reviews, train_scores, test_scores = train_test_split(
    reviews, scores, test_size=0.2, random_state=42
)


###################################

# 활성화 함수 정의
def sigmoid(x):     # 은닉층
    return 1 / (1 + np.exp(-x))

def softmax(a):     # 출력층
    exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

###################################

# 신경망 초기화 함수   --> 각 테스트에 대해 맨 처음 정의 시
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

# 순전파 함수 --> 예측값 도출
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

# 손실 함수 (교차 엔트로피 오차) --> 예측값 & 실제값 비교
def cross_entropy_error(y, t):
    if isinstance(t, list):
        t = np.array(t)  # 리스트라면 NumPy 배열로 변환

    if t.ndim == 1:  # 정수 레이블을 원-핫 인코딩으로 변환
        t = np.eye(y.shape[1])[t - 1]

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t.argmax(axis=1)] + 1e-7)) / batch_size

####################################

def loss(network, x, t):                                 # 모델의 예측 결과값을 받아 실제 정답과 오차 계산 (예측 & 평가 한 번에 실행)
    y = forward(network, x)
    return cross_entropy_error(y,t)

def develop_gradient(network, x, t):            # 매개변수 갱신
    loss_W = lambda W: loss(network, x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, network['W1'])
    grads['b1'] = numerical_gradient(loss_W, network['b1'])
    grads['W2'] = numerical_gradient(loss_W, network['W2'])
    grads['b2'] = numerical_gradient(loss_W, network['b2'])
    grads['W3'] = numerical_gradient(loss_W, network['W3'])
    grads['b3'] = numerical_gradient(loss_W, network['b3'])

    return grads


def numerical_gradient(f, x):   # 손실함수의 기울기 도출 -> 이후 이를 바탕으로 매개변수 갱신
    h = 1e-4
    grad = np.zeros_like(x)     # x와 형상이 같은 배열을 생성

    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)

        x[idx] = tmp_val  # 값 복원
        it.iternext()

    return grad

# 경사하강법 
def gradient_descent(f, init_x, lr=0.01, step_num=30):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x

#################

# 텍스트 토큰화 및 패딩
tokenizer = Tokenizer(num_words=2000)  # 상위 2000개 단어만 사용
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

###############################################


# 정확도 평가 함수
def predict_scores(network, x):
    y = forward(network, x)
    predicted_scores = np.argmax(y, axis=1) + 1
    return predicted_scores

def calculate_accuracy(network, x, actual_scores):
    predicted_scores = predict_scores(network, x)
    return np.sum(predicted_scores == actual_scores)

#################################################

# 실행 함수 정의
def perform(network, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=100):

    accuracies = []

    for epoch in range(epochs):
        learning_rate = 0.01

        # 각 에포크마다 전체 훈련 데이터를 사용
        for i in range(0, len(train_data), batch_size):
            x_batch = train_data[i:i+batch_size]
            t_batch = train_labels[i:i+batch_size]

            # t_batch를 NumPy 배열로 변환
            t_batch = np.array(t_batch)

            # 기울기 계산
            grads = develop_gradient(network, x_batch, t_batch)

            # print(f"{epoch}, {i}/{len(train_data)}")

            # 경사 하강법으로 매개변수 갱신
            for key in ('W1', 'b1', 'W2', 'b2', 'W3', 'b3'):
                network[key] -= learning_rate * grads[key]

        # 에포크마다 테스트 데이터셋으로 정확도 평가
        test_accuracy = 0
        for i in range(0, len(test_data), batch_size):
            x_test_batch = test_data[i:i+batch_size]
            t_test_batch = test_labels[i:i+batch_size]

            y_test_batch = forward(network, x_test_batch)
            test_accuracy += np.sum(np.argmax(y_test_batch, axis=1) == np.argmax(t_test_batch, axis=1))

        test_accuracy /= len(test_data)
        accuracies.append(test_accuracy)
        print(f"Epoch {epoch + 1}, Test Accuracy: {test_accuracy}")

    mean_accuracy = np.mean(accuracies)
    return mean_accuracy

    
# 모델 초기화 및 학습
network = initialize_network(input_size, hidden_size1, hidden_size2, output_size)
mean_accuracy = perform(network, train_data, train_labels, test_data, test_labels, epochs=10, batch_size=100)
print("Mean Accuracy:", mean_accuracy)

'''
Epoch 1, Test Accuracy: 0.186
Epoch 2, Test Accuracy: 0.186
Epoch 3, Test Accuracy: 0.186
Epoch 4, Test Accuracy: 0.186
Epoch 5, Test Accuracy: 0.186
Epoch 6, Test Accuracy: 0.186
Epoch 7, Test Accuracy: 0.186
Epoch 8, Test Accuracy: 0.186
Epoch 9, Test Accuracy: 0.186
Epoch 10, Test Accuracy: 0.186

Mean Accuracy: 0.186
'''