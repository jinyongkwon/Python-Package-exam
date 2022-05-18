import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 도미와 빙어데이터 준비
bream_length = pd.read_csv("bream_length.csv", sep='\t', names=["a"])
bream_weight = pd.read_csv("bream_weight.csv", sep='\t', names=["a"])
smelt_length = pd.read_csv("smelt_length.csv", sep='\t', names=["a"])
smelt_weight = pd.read_csv("smelt_weight.csv", sep='\t', names=["a"])

# 눈으로 확인
plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
# plt.show()

# 데이터 전처리
fish_length = np.concatenate((bream_length.values, smelt_length.values))
fish_weight = np.concatenate((bream_weight.values, smelt_weight.values))

fish_data = np.column_stack((fish_length, fish_weight))
fish_target = [1] * 35 + [0] * 14

input_arr = np.array(fish_data)
target_arr = np.array(fish_target)

# 크로스 밸리데이션
index = np.arange(49)
np.random.shuffle(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]
test_input = input_arr[index[35:]]
test_target = target_arr[index[35:]]

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()
