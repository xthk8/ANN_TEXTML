import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# 데이터 읽어오기
def read_reviews(base_path):
    data = []
    labels = []
    for score in range(1, 6):
        folder_path = os.path.join(base_path, str(score))
        # glob 모듈의 glob 함수는 사용자가 제시한 조건에 맞는 파일명을 리스트 형식으로 반환
        for file_path in glob.glob(os.path.join(folder_path, '*.txt')):
            with open(file_path, 'r', encoding='utf8') as file:
                review_text = file.read().strip()
                data.append(review_text)
                labels.append(score)
    return data, labels

# 파일 경로 정의 및 함수 호출
review_path = "C:/Users/USER/Desktop/학부연구/밑바닥부터 시작하는 딥러닝/reviews3"
reviews, scores = read_reviews(review_path) # 리뷰 텍스트 & 점수 저장

# 텍스트 토큰화 및 패딩
'''
Tokenizer 클래스
- 텍스트 토큰화 : 문자열을 분석하기 쉬운 형태로 변환
- 텍스트를 개별 단어 또는 토큰으로 분리하고, 각 단어에 고유한 정수 ID 부여
=> 텍스트 데이터를 신경망이 이해할 수 있는 형태로 변환

- num_words : 가장 자주 등장하는 개수만큼의 단어만 유지
- fit_on_texts : 토큰화를 위한 내부 사전 구축
- texts_to_sequences : 앞에서 구축된 사전을 사용하여 텍스트를 정수 시퀀스로 변환

- pad_sequences : 시퀀스 데이터(정수 시퀀스)의 길이를 동일하게 맞춤
  => 신경망의 입력 샘플 길이를 동일화
  -> 매개변수
    1) sequences : 정수 시퀀스의 리스트
    2) maxlen : 최대 길이, 이 값보다 길면 잘려나가고 짧으면 지정된 방식에 따라 패딩
    3) dtype : 출력 시퀀스의 자료형, 기본적으로 int32
    4) padding : pre/post를 지정하여 앞이나 뒤에 패딩을 삽입할지 결정, 기본값은 pre
    5) truncating : pre/post를 지정하여 시퀀스가 maxlen보다 길 때 앞이나 뒤를 잘라낼지 결정, 기본값은 pre
'''
tokenizer = Tokenizer(num_words=100)               # 가장 빈번한 2000(최대 단어 수 지정)개 단어만 사용
tokenizer.fit_on_texts(reviews)                     # 각 리뷰 데이터들에서 단어를 토큰화하고, 각 단어에 고유한 인덱스를 할당
sequences = tokenizer.texts_to_sequences(reviews)   # 텍스트를 이러한 인덱스의 시퀀스로 변환 => 신경망 모델의 입력
maxlen = 25
data = pad_sequences(sequences, maxlen=maxlen)



# 원-핫 인코딩
'''
OneHotEncoder 라이브러리
- categories : 인코더가 데이터에서 자동으로 범주를 찾도록 지정 => 데이터에서 발견되는 모든 고유한 값들이 범주로 사용됨
- handle_unknown : 인코더가 학습 단계에서 보지 못한 범주를 만났을 때 무시하도록 설정

encoder.fit (인코더 학습)
- scores 배열의 유니크한 값들을 학습
- 1차원 배열일 가능성이 높으므로 reshape을 사용하여 2차원 배열로 변환
- 여기서 -1 : 해당 차원의 크기를 자동으로 계산

encoder.transform (데이터 변환)
- scores 데이터를 원-핫 인코딩된 형태로 변환
- 각 행은 원본 scores 배열의 각 요소에 대응하는 원-핫 벡터
- toarray() : 넘파이 배열로 전환 (원-핫 인코딩된 결과는 기본적으로 희소 행렬 형태)

1 → [1, 0, 0, 0, 0]
2 → [0, 1, 0, 0, 0]
3 → [0, 0, 1, 0, 0]
4 → [0, 0, 0, 1, 0]
5 → [0, 0, 0, 0, 1]

'''
encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')
encoder.fit(np.array(scores).reshape(-1, 1))
labels = encoder.transform(np.array(scores).reshape(-1, 1)).toarray()

########################################################################

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

# 교차 엔트로피 오차 손실값
def loss(network, x, t):
    y = forward(network, x)
    return cross_entropy_error(y, t)

# 기울기 계산
def numerical_gradient(f, x):
    h = 1e-4                    # 반올림오차
    grad = np.zeros_like(x)     # x와 동일한 배열 형태 생성 (0으로 채워짐)

    # np.nditer : 넘파이 배열의 요소들을 효율적으로 순회
    # (반복할 배열, 반복자의 동작(여기서는 현재 요소의 다차원 인덱스 제공), 각 연산에 대한 플래크(여기서는 반복 중에 배열의 요소 수정 가능)) 
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h     
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)
        grad[idx] = (fxh1 - fxh2) / (2 * h)     # 중심(중앙)차분

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

    # 데이터 분할 (교차검증)
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
        for i in range(0, len(test_data), batch_size):
            x_test_batch = test_data[i:i+batch_size]
            t_test_batch = test_labels[i:i+batch_size]

            y_test_batch = forward(network, x_test_batch)
            test_accuracy = np.sum(np.argmax(y_test_batch, axis=1) == np.argmax(t_test_batch, axis=1))

        test_accuracy /= len(test_data)
        print(test_accuracy)
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