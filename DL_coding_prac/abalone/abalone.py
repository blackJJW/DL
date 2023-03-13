import numpy as np # 배열 및 수치 연산을 지원
import csv         # 아발로니 데이터셋 파일을 읽는 데 필요한 csv 모듈
import time        # 난수 함수 초기화에 이용할 time 모듈

np.random.seed(1234) # 난수 발생 패턴을 고정

def randomize():     # 현재 시간을 이용해 난수 발생 패턴을 다시 설정
    np.random.seed(time.time()) 

# Hyper Parameters -----
RND_MEAN = 0     # 정규분포 난숫값의 평균값
RND_STD = 0.0030 # 정규분포 난숫값의 표준편차

LEARNING_RATE = 0.001 # 학습에 매우 커다란 영향을 미치는 중요한 하이퍼파라미터 : 학습률
#----------------------

# 실험용 메인 함수
# epoch_count : 학습 횟수, mb_size : 미니배치 크기, report : 중간 보고 주기
# 매개 변수에 디폴트값을 지정함으로써 함수를 호출할 때 인수를 생략 가능
def abalone_exec(epoch_count = 10, mb_size = 10, report = 1):
    load_abalone_dataset()                       # 데이터셋을 읽어들임
    init_model()                                 # 모델의 파라미터들을 초기화
    train_and_test(epoch_count, mb_size, report) # 학습 및 평가 과정을 수행

# 데이터 적재 함수 정의
# 데이터셋 파일 내용을 메모리로 읽어들여 이용할 수 있게 준비
def load_abalone_dataset():
    with open('../data/abalone.csv') as csvfile: 
        csvreader = csv.reader(csvfile) # 메모리로 읽어들임
        next(csvreader, None)           # next() 함수 호출은 파일의 첫 행을 읽기 않고 건너뜀
        rows = []             
        for row in csvreader:           # 파일 각 행에 담긴 전복 개체별 정보를 csvreader 객체를 이용해 rows 리스트에 수집
            rows.append(row)

    global data, input_cnt, output_cnt  # data와 함께 전역 변수로 선언되어 다른 함수에서도 이용 가능
    input_cnt, output_cnt = 10, 1       # 전복 개체들의 입출력 벡터 정보를 저장할 data 행렬을 만들 때 크기 지정에 이용
    data = np.zeros([len(rows), input_cnt + output_cnt]) 

    for n, row in enumerate(rows):
        # 비선형인 성별 정보를 원-핫 벡터 표현으로 변환하는 처리
        if row[0] == 'I': data[n, 0] = 1
        if row[0] == 'M': data[n, 1] = 1
        if row[0] == 'F': data[n, 2] = 1
        data[n, 3:] = row[1:] # 성별 이외의 정보 항목들을 일괄 복제 


# 파라미터 초기화 함수 정의

# 파라미터를 초기화
# 메인 함수인 abalone_exec()에서 호출됨
# 단층 퍼셉트론의 가중치 파라미터 weight와 편향 파라미터 bias를 초기화
# load_abalone_dataset() 함수에서 input_cnt, output_cnt 변수에 지정한 입출력 벡터 크기를 이용해 가중치 행렬과 편향 벡터의 크기를 결정
def init_model():
    global weight, bias, input_cnt, output_cnt # 전역 변수로 선언
    # 가중치 행렬 값들은 np.random.normal() 함수를 이용해 정규분포를 갖는 난숫값으로 초기화
    # 경사하강법의 출발점에 해당하는 파라미터의 초깃값을 실행할 때마다 달라지게 만들려는 의도
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    # 편향은 초기에 지나친 영향을 주어 학습에 역효과를 불러오지 않도록 0으로 초기화
    bias = np.zeros([output_cnt])

# 학습 및 평가 함수 정의
# 학습과 평가 과정을 일괄 실행
# 적재된 데이터셋과 초기화된 파라미터를 이용하여 학습 및 평가의 전체 과정을 수행
# 이중 반복을 이용, epoch_count 인수로 지정된 에포크 수 만큼 학습 을 반복
# 반복문 안에서 다시 step_count값 만큼 미니배치 처리를 반복
def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size) # 데이터를 뒤섞고 학습용 데이터셋과 평가용 데이터셋을 분리하는 등의 데이터 정렬 작업도 수행
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses, accs = [], []

        for n in range(step_count): 
            train_x, train_y = get_train_data(mb_size, n) # 학습용 미니배치 데이터를 얻어와 run_train() 함수로 학습시키는 방식으로 처리
            loss, acc = run_train(train_x, train_y)
            # 미니배치 단위에서의 비용과 정확도를 보고받아 리스트 변수 losses와 accs에 집계
            losses.append(loss)
            accs.append(acc)
        
        # 각 에포크가 끝나면 report 인수로 지정된 보고 주기에 해당하는 지 검사
        # 해당되면 중간 평사 함수 run_test()를 호출한 후 그 결과를 출력
        # 이때 학습 과정에서 집계한 손실 함수와 정확도의 평균값도 함께 출력
        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss = {:5.3f}, accuracy = {:5.3f}/{:5.3f}'.format(epoch + 1, np.mean(losses), np.mean(accs), acc))

    # 전체 에포크 처리가 끝나면 다시 최종 평가 함수 run_test()를 호출하고 그 결과를 출력
    # 중간 평가와 최종 평가에 동일한 평가용 데이터셋을 반복적으로 이용하기 때문에 이를 이중 반복 실행 전에 미리 get_test_data() 함수를 호출해  test_x, test_y에 저장해두어 반복해서 활용
    final_acc = run_test(test_x, test_y)
    print('\nFinal Test : final accuracy = {:5.3f}'.format(final_acc))

# 학습 및 평가 데이터 획득 함수 정의
# train_and_test() 함수가 학습 및 평가 데이터를 얻을 때 호출하는 세 함수를 정의
# arrange_data() 함수는 train_and_test() 함수가 이중 반복을 준비할 때 단 한 번 호출
def arrange_data(mb_size): 
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arrange(data.shape[0]) # 데이터 수만큼의 일련번호를 발생
    np.random.shuffle(shuffle_map)          # 무작위로 순서를 섞는다.
    step_count = int(data.shape[0] * 0.8) // mb_size 
    test_begin_idx = step_count * mb_size
    return step_count

# 평가 데이터를 일괄 공급
def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    # 각 행에 대해 뒤에서 output_cnt 번째 되는 원소 위치를 기준으로 분할해 앞 쪽을 입력 벡터, 뒷 쪽을 정답 벡터로 반황
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]

# 학습 데이터를 공급
# get_test_dat() 함수와 비슷한 처리를 수행
def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0: # 각 에포크 첫 번재 호출, nth값이 0일 때에 한하여 학습 데이터 부분에 대한 부분적인 순서를 뒤섞어 에포크마다 다른 순서로 학습이 수행
        np.random.shuffle(shuffle_map[:test_begin_idx])

    # 미니배치 구간의위치를 따져 그 구간에 해당하는 suffle_map이 가리키는 데이터들만 반환
    train_data = data[shuffle_map[mb_size * nth : mb_size * (nth + 1)]]

    # 반환하는 각 행에 대해 입력 벡터 부분과 정답 벡터 부분을 분할해 반환
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]

# 학습 실행 함수와 평가 실행 함수 정의
# train_and_test() 함수가 호출하는 학습 실행 함수와 평가 실행 함수를 정의
# 미니배치 학습을 처리를 담당
# 학습용 데이터의 일부로 주어지는 미니배치 입력 행렬 x와 정답 행렬 y를 이용해 한 스텝의 학습을 수행
def run_train(x, y):
    # 순전파 처리
    output, aux_nn = forward_neuralnet(x)      # forward_neuralnet() 함수가 단층 퍼셉트론 신경망에 대한 순전파를 수행, 입력 행렬 x로부터 신경망 출력 output을 구한다.
    loss, aus_pp = forward_postproc(output, y) # forward_postproc() 함수가 회귀 분석 문제의 성격에 맞춘 후처리 순전파 작업을 수행해 output과 y로부터 손실 함수 loss를 계산
    
    # 보고용 정확도 계산
    accuracy = eval_accuracy(output, y)

    G_loss = 1.0

    # 역전파 처리
    # G_loss로 부터 G_output을 구한다.
    # 순전파 함수가 역전파용 보조 정보로서 보고했던 aux_pp 제공
    G_output = backprop_postproc(G_loss, aux_pp)
    backprop_neuralnet(G_output, aux_nn)

    return loss, accuracy

# 평가 데이터 전체에 대해 일괄적으로 평가를 수행
def run_test(x, y):
    output, _ = forward_neuralnet(x)    # 신경망 부분에 대한 순전파 처리만 수행
    accuracy = eval_accuracy(output, y) # 정확도를 계산해서 반환

    return accuracy

# 단층 퍼셉트론에 대한 순전파 및 역전파 함수 정의
# 단층 퍼셉트론 신경망 부분에 대해 순전파 처리와 역전파 처리를 수행하는 두 함수를 정의
def forward_neuralnet(x):
    global weight, bias
    output = np.matmul(x, weight) + bias # 입력 행렬 x에 대해 가중치 weight를 곱하고 편향 벡터 bias를 더하는 간단한 방법으로 신경망 출력에 해당하는 output 행렬을 생성
    # 가중치 곱셈은 행렬끼리의 곱셈, 편향 덧셈은 행렬과 벡터의 덧셈

    return output, x

# 역전파 처리를 수행
def backprop_neuralnet(G_output, x):
    global weight, bias
    g_output_w = x.transpose()
    
    # G_w, G_b : weight, bias 성분의 손실 기울기
    G_w = np.matmul(g_output_w, G_output)
    G_b = np.sum(G_output, axis = 0)

    weight -= LEARNING_RATE * G_w
    bias -= LEARNING_RATE * G_b
