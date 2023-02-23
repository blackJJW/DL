import numpy as np
import csv
import time

np.random.seed(1234)

def randomize():
    np.random.seed(time.time())

# Hyper Parameters -----
RND_MEAN = 0
RND_STD = 0.0030

LEARNING_RATE = 0.001
#----------------------

def abalone_exec(epoch_count = 10, mb_size = 10, report = 1):
    load_abalone_dataset()
    init_model()
    train_and_test(epoch_count, mb_size, report)

def load_abalone_dataset():
    with open('../data/abalone.csv') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader, None)
        rows = []
        for row in csvreader:
            rows.append(row)

    global data, input_cnt, output_cnt
    input_cnt, output_cnt = 10, 1
    data = np.zeros([len(rows), input_cnt + output_cnt])

    for n, row in enumerate(rows):
        if row[0] == 'I': data[n, 0] = 1
        if row[0] == 'M': data[n, 1] = 1
        if row[0] == 'F': data[n, 2] = 1
        data[n, 3:] = row[1:]

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
