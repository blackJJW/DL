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
def init_model():
    global weight, bias, input_cnt, output_cnt
    weight = np.random.normal(RND_MEAN, RND_STD, [input_cnt, output_cnt])
    bias = np.zeros([output_cnt])

def train_and_test(epoch_count, mb_size, report):
    step_count = arrange_data(mb_size)
    test_x, test_y = get_test_data()

    for epoch in range(epoch_count):
        losses, accs = [], []

        for n in range(step_count):
            train_x, train_y = get_train_data(mb_size, n)
            loss, acc = run_train(train_x, train_y)
            losses.append(loss)
            accs.append(acc)

        if report > 0 and (epoch + 1) % report == 0:
            acc = run_test(test_x, test_y)
            print('Epoch {}: loss = {:5.3f}, accuracy = {:5.3f}/{:5.3f}'.format(epoch + 1, np.mean(losses), np.mean(accs), acc))

    final_acc = run_test(test_x, test_y)
    print('\nFinal Test : final accuracy = {:5.3f}'.format(final_acc))

# 학습 및 평가 데이터 획득 함수 정의
def arrange_data(mb_size):
    global data, shuffle_map, test_begin_idx
    shuffle_map = np.arrange(data.shape[0])
    np.random.shuffle(shuffle_map)
    step_count = int(data.shape[0] * 0.8) // mb_size
    test_begin_idx = step_count * mb_size
    return step_count

def get_test_data():
    global data, shuffle_map, test_begin_idx, output_cnt
    test_data = data[shuffle_map[test_begin_idx:]]
    return test_data[:, :-output_cnt], test_data[:, -output_cnt:]

def get_train_data(mb_size, nth):
    global data, shuffle_map, test_begin_idx, output_cnt
    if nth == 0:
        np.random.shuffle(shuffle_map[:test_begin_idx])
    train_data = data[shuffle_map[mb_size * nth : mb_size * (nth + 1)]]
    return train_data[:, :-output_cnt], train_data[:, -output_cnt:]
