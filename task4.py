"""
CS6140 Project 4
@filename: task4.py
@author: Haotian Shen, Qiaozhi Liu
"""
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.core.display_functions import display
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Net1(nn.Module):
    def __init__(self, dim):
        super(Net1, self).__init__()
        self.fc1 = nn.Linear(dim, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class Net2(nn.Module):
    def __init__(self, dim):
        super(Net2, self).__init__()
        self.fc1 = nn.Linear(dim, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 2)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


class Net3(nn.Module):
    def __init__(self, dim):
        super(Net3, self).__init__()
        self.fc1 = nn.Linear(dim, 6)
        self.dropout1 = nn.Dropout(p=0.05)
        self.fc2 = nn.Linear(6, 4)
        self.dropout2 = nn.Dropout(p=0.05)
        self.fc3 = nn.Linear(4, 2)
        self.dropout3 = nn.Dropout(p=0.05)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x


class Net4(nn.Module):
    def __init__(self, dim):
        super(Net4, self).__init__()
        self.fc1 = nn.Linear(dim, 4)
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


class Net5(nn.Module):
    def __init__(self, dim):
        super(Net5, self).__init__()
        self.fc1 = nn.Linear(dim, 6)
        self.fc2 = nn.Linear(6, 4)
        self.fc3 = nn.Linear(4, 2)
        self.fc4 = nn.Linear(2, 1)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.sigmoid(self.fc4(x))
        return x


def read_csv(filename: str) -> DataFrame:
    """Reads the csv file into a pandas DataFrame
    :param filename: str.
    :return: a pandas DataFrame.
    """
    df = pd.read_csv(filename)
    return df


def flip_bits(num: int, length: int) -> List:
    """Flips each and every bit of a binary representation of the provided length to generate one-hot encoding.
    For example, if the input is 0000, the output will be a list containing 1000, 0100, 0010, 0001 in integer format.
    :param num: the binary representation of 0.
    :param length: indicates the length of a binary representation of 0.
    :return: a list of integers derived from the binary vector (one-hot encoding).
    """
    result = []
    for i in range(length):
        flipped = int(str(num), 2) ^ (1 << i)
        result.append(flipped)
    return result


def one_hot_encoding(df: DataFrame) -> Dict:
    """Transforms the categorical data using one-hot encoding.
    The categorical value will be transformed into an integer representation of its binary encoding.
    :param df: the raw un-transformed pandas DataFrame.
    :return: a dictionary that records the transformation rules.
    """
    category_dict = {}
    for i in range(len(df.columns)):
        try:
            float(df.iloc[0, i])
        except ValueError:
            cat_set = set(df.iloc[:, i])
            num_cat = len(cat_set)
            print(f"The number of categories in this feature is {num_cat}")
            print(f"Features are {cat_set}")
            cur_dict = {}
            index = 0
            binary_rep = flip_bits(0, num_cat)
            for cat in cat_set:
                cur_dict[cat] = binary_rep[index]
                index += 1
            category_dict[i] = cur_dict
    return category_dict


def clean_data(df: DataFrame) -> DataFrame:
    """Cleans the raw data and turn all columns into numerical values.
    The categorical value will be transformed using one-hot encoding.
    :param df: the raw input pandas DataFrame.
    :return: a cleaned version of pandas DataFrame.
    """
    features = df.columns
    category_dict = one_hot_encoding(df)
    num_rows = len(df)
    num_cols = len(df.columns)
    cleaned_data = []
    for i in range(num_rows):
        cur_row = []
        for j in range(num_cols):
            if j in category_dict:
                cur_row.append(category_dict[j][df.iloc[i, j]])
            else:
                cur_row.append(float(df.iloc[i, j]))
        cleaned_data.append(cur_row)
    return pd.DataFrame(cleaned_data, columns=features)


def normalize_data(df: DataFrame) -> DataFrame:
    """Normalizes the dataset on the scale of standard deviation.
    :param df: the cleaned pandas DataFrame
    :return: normalized data
    """
    A = df.values
    m = np.mean(A, axis=0)
    D = A - m
    std = np.std(D, axis=0)
    D = D / std
    D = pd.DataFrame(D, columns=df.columns)
    return D


def main(network: nn.Module, index: int):
    train_df = read_csv('data/heart_disease_data/heart_train_718.csv')
    test_df = read_csv('data/heart_disease_data/heart_test_200.csv')
    print(train_df.shape, test_df.shape)

    # combine the training and test set to prepare for data normalization
    combined_df = pd.concat([train_df, test_df], axis=0)
    combined_df = clean_data(combined_df)

    # separate the cleaned data into X and y
    X, y = combined_df.drop('HeartDisease', axis=1), combined_df['HeartDisease']
    # normalize the data
    X_normalized = normalize_data(X)
    X_train = X_normalized.iloc[:len(train_df), :]
    X_test = X_normalized.iloc[len(train_df):, :]
    y_train = y.iloc[:len(train_df)]
    y_test = y.iloc[len(train_df):]

    X_train = X_train.drop(['Sex', 'FastingBS', 'RestingECG'], axis=1)
    X_test = X_test.drop(['Sex', 'FastingBS', 'RestingECG'], axis=1)

    print(X_train.shape)
    display(X_train.head())
    print(X_test.shape)

    optimizer = optim.Adam(network.parameters(), lr=learning_rate)
    criterion = nn.BCELoss()

    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f'Train epoch: {epoch} [{batch_idx * len(inputs)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                train_losses.append(loss.item())
                train_counter.append(batch_idx * train_batch_size + (epoch - 1) * len(train_loader.dataset))
                torch.save(network.state_dict(), f'results/task4_network{index}_model.pth')
                torch.save(optimizer.state_dict(), f'results/task4_network{index}_optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = network(inputs)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * inputs.size(0)
                pred = (outputs > 0.5).float()
                correct += (pred == labels).sum().item()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.2f}%)\n')
        return 100. * correct / len(test_loader.dataset)

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    plt.figure()
    plt.plot(train_counter, train_losses, color='b')
    plt.scatter(test_counter, test_losses, color='r')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(f'images/task4_performance_with_100_epochs_network{index}')
    plt.show()

    return test()


if __name__ == '__main__':
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    n_epochs = 200
    log_interval = 10
    train_batch_size = 64
    test_batch_size = 100
    learning_rate = 0.001
    input_dim = 8
    net1 = Net1(input_dim)
    net2 = Net2(input_dim)
    net3 = Net3(input_dim)
    net4 = Net4(input_dim)
    net5 = Net5(input_dim)
    network_dict = {1: net1, 2: net2, 3: net3, 4: net4, 5: net5}
    network_acc_dict = {}
    input_size = (train_batch_size, input_dim)
    for index in network_dict:
        net = network_dict[index]
        print(summary(net, input_size=input_size, verbose=0))
        print(net)
        acc = main(net, index)
        network_acc_dict[index] = acc
    print(network_acc_dict)
