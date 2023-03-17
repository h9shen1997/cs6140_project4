"""
CS6140 Project 4
@filename: task2.py
@author: Haotian Shen, Qiaozhi Liu
"""
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib import pyplot as plt

conv_filter_default_step = 2


class CustomNet(nn.Module):
    def __init__(self, num_conv_layers=2, kernel_size=3, num_init_conv_filter=10, dropout_rate=0.25, padding=1):
        super(CustomNet, self).__init__()
        self.num_conv_layers = num_conv_layers
        self.kernel_size = kernel_size
        self.num_init_conv_filter = num_init_conv_filter
        self.padding = padding
        self.dropout = dropout_rate
        for i in range(num_conv_layers):
            conv_name = f'conv{i + 1}'
            if i == 0:
                setattr(self, conv_name,
                        nn.Conv2d(1, self.num_init_conv_filter, kernel_size=self.kernel_size, padding=self.padding))
            else:
                setattr(self, conv_name, nn.Conv2d(self.num_init_conv_filter * (conv_filter_default_step ** (i - 1)),
                                                   self.num_init_conv_filter * (conv_filter_default_step ** i),
                                                   kernel_size=self.kernel_size, padding=self.padding))
        self.conv2_drop = nn.Dropout2d(p=self.dropout)
        output_shape = 28
        for _ in range(self.num_conv_layers):
            output_shape = ((output_shape + padding * 2 - kernel_size) + 1) // 2
        linear_input_shape = output_shape * output_shape * self.num_init_conv_filter * (2 ** (self.num_conv_layers - 1))
        self.linear_input_shape = linear_input_shape
        self.fc1 = nn.Linear(linear_input_shape, linear_input_shape // 6)
        self.fc2 = nn.Linear(linear_input_shape // 6, 10)

    def forward(self, x):
        # convolution layers
        for i in range(self.num_conv_layers):
            conv_name = f'conv{i + 1}'
            x = getattr(self, conv_name)(x)
            if i != 0:
                x = self.conv2_drop(x)
            x = F.relu(x)
            x = F.max_pool2d(x, 2)

        # fully connected layers
        x = x.view(-1, self.linear_input_shape)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


def train_with_param(num_conv_layer, kernel_size, num_init_conv_filter, dropout_rate):
    train_dataset = torchvision.datasets.FashionMNIST(root='./files/', train=True, download=True,
                                                      transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.FashionMNIST(root='./files/', train=False, download=True,
                                                     transform=torchvision.transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size_test, shuffle=True)

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    network = CustomNet(num_conv_layers=num_conv_layer, kernel_size=kernel_size,
                        num_init_conv_filter=num_init_conv_filter, dropout_rate=dropout_rate)
    print(network)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
                train_losses.append(loss.item())
                train_counter.append(batch_idx * 100 + (epoch - 1) * len(train_loader.dataset))
                torch.save(network.state_dict(), 'results/task2_model.pth')
                torch.save(optimizer.state_dict(), 'results/task2_optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print(
            f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({100. * correct / len(test_loader.dataset):.0f}%)\n')
        return 100. * correct / len(test_loader.dataset)

    acc = test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        acc = test()

    plt.figure()
    plt.plot(train_counter, train_losses, color='b')
    plt.scatter(test_counter, test_losses, color='r')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig(f'images/task2_performance_with_30_epochs')
    plt.show()

    with torch.no_grad():
        output = network(example_data)
    plt.figure()
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f'Prediction: {output.data.max(1, keepdim=True)[1][i].item()}')
        plt.xticks([])
        plt.yticks([])
    plt.savefig('images/task2_fashion_outputs_with_prediction_label')
    plt.show()

    return acc


def find_best_param():
    param_dict = {
        'num_conv_layer': [2],  # num_conv_layer
        'kernel_size': [2, 3, 4, 5],  # kernel_size
        'num_init_conv_filter': [8, 16, 32],  # num_init_conv_filter
        'dropout': [0.1, 0.25, 0.5]  # dropout
    }

    highest_acc = 0
    best_param = {}
    for values in itertools.product(*param_dict.values()):
        # values is a tuple of the parameter values for one combination
        print(values)
        num_conv_layer, kernel_size, num_init_conv_filter, dropout = values

        cur_acc = train_with_param(num_conv_layer=num_conv_layer, kernel_size=kernel_size,
                                   num_init_conv_filter=num_init_conv_filter, dropout_rate=dropout)
        acc_dict[values] = cur_acc

        if cur_acc > highest_acc:
            best_param = {
                'num_conv_layer': num_conv_layer,
                'kernel_size': kernel_size,
                'num_init_conv_filter': num_init_conv_filter,
                'dropout': dropout
            }
            highest_acc = cur_acc
    return highest_acc, best_param


if __name__ == '__main__':
    n_epochs = 30
    batch_size_train = 100
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    acc_dict = {}
    # highest_acc, best_param = find_best_param()
    # print(highest_acc)
    # print(best_param)
    train_with_param(2, 2, 32, 0.25)
