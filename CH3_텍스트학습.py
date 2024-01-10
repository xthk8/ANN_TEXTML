# 맨 처음 - 데이터 읽어오기

import os
import glob

def read_reviews_from_folders(base_path):
    data = []
    labels = []
    for score in range(1, 6):  
        folder_path = os.path.join(base_path, str(score))
        # 각각의 점수에 해당하는 텍스트 파일 읽기
        for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
            with open(file_path, 'r', encoding='utf8') as file:
                review_text = file.read().strip()
                data.append(review_text)
                labels.append(score)
    return data, labels

review_base_path = 'C:/Users/USER/Desktop/학부연구/밑바닥부터 시작하는 딥러닝/Review/reviews' 

reviews, scores = read_reviews_from_folders(review_base_path)

###################################

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import random

# 활성화 함수 정의
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    exp_a = np.exp(a - np.max(a, axis=1, keepdims=True))
    sum_exp_a = np.sum(exp_a, axis=1, keepdims=True)
    y = exp_a / sum_exp_a
    return y

# 신경망 초기화 함수
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

'''
def loss(x, t):
    y = forward(x)

def numerical_gradient(x, t):
    loss_W = lambda W: loss(x, t)

    grads = {}
    grads['W1'] = numerical_gradient(loss_W, network['W1'])
    grads['b1'] = numerical_gradient(loss_W, network['b1'])
    grads['W2'] = numerical_gradient(loss_W, network['W2'])
    grads['b2'] = numerical_gradient(loss_W, network['b2'])

    return grads
'''

#################


# 데이터셋 분할
train_reviews, test_reviews, train_scores, test_scores = train_test_split(
    reviews, scores, test_size=0.2, random_state=42
)

# 텍스트 토큰화 및 패딩
tokenizer = Tokenizer(num_words=2000)  # 상위 2000개 단어만 사용
tokenizer.fit_on_texts(train_reviews)
train_sequences = tokenizer.texts_to_sequences(train_reviews)
test_sequences = tokenizer.texts_to_sequences(test_reviews)

# 모든 시퀀스를 같은 길이로 맞춤 (예: 100 단어)
maxlen = 100
train_data = pad_sequences(train_sequences, maxlen=maxlen)
test_data = pad_sequences(test_sequences, maxlen=maxlen)

# 원-핫 인코딩 적용
encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
encoder.fit(train_data)
train_vectors = encoder.transform(train_data).toarray()
test_vectors = encoder.transform(test_data).toarray()

# 신경망 구조 설정 및 초기화
input_size = 100
hidden_size1 = 50
hidden_size2 = 5
output_size = 5

network = initialize_network(input_size, hidden_size1, hidden_size2, output_size)

# 정확도 평가 함수
def predict_scores(network, x):
    y = forward(network, x)
    predicted_scores = np.argmax(y, axis=1) + 1
    return predicted_scores

def calculate_accuracy(network, x, actual_scores):
    predicted_scores = predict_scores(network, x)
    return np.mean(predicted_scores == actual_scores)

# 랜덤 샘플링 및 평가
def random_sample_and_evaluate(network, train_data, train_scores, test_data, test_scores, num_samples=30, sample_size=100):
    accuracies = []

    for _ in range(num_samples):
        # 학습 데이터셋에서 랜덤하게 샘플링
        train_indices = np.random.choice(len(train_data), sample_size, replace=False)
        sampled_train_data = train_data[train_indices]
        sampled_train_scores = np.array(train_scores)[train_indices]

        # 테스트 데이터셋에서 랜덤하게 샘플링
        test_indices = np.random.choice(len(test_data), sample_size, replace=False)
        sampled_test_data = test_data[test_indices]
        sampled_test_scores = np.array(test_scores)[test_indices]

        # 이 부분에서 모델을 학습시킬 수 있습니다.
        # 예: train_model(network, sampled_train_data, sampled_train_scores)

        # 테스트 데이터셋으로 정확도 계산
        accuracy = calculate_accuracy(network, sampled_test_data, sampled_test_scores)
        accuracies.append(accuracy)

    # 평균 정확도 계산
    mean_accuracy = np.mean(accuracies)
    return mean_accuracy

# 랜덤 샘플링 및 평가
mean_accuracy = random_sample_and_evaluate(network, train_data, train_scores, test_data, test_scores)
print("Mean Accuracy over 30 samples:", mean_accuracy)