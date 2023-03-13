import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision


def main():
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=True, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_train, shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('files/', train=False, download=True,
                                   transform=torchvision.transforms.Compose([
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                           (0.1307,), (0.3081,))
                                   ])),
        batch_size=batch_size_test, shuffle=True
    )

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    plt.figure()
    for i in range(20):
        plt.subplot(4, 5, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'images/task1_example_outputs')
    plt.show()

    network = Net()
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
                train_counter.append(batch_idx * 64 + (epoch - 1) * len(train_loader.dataset))
                torch.save(network.state_dict(), 'results/task1_model.pth')
                torch.save(optimizer.state_dict(), 'results/task1_optimizer.pth')

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
    plt.savefig(f'images/task1_performance_with_3_epochs')
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
    plt.savefig('images/task1_example_outputs_with_prediction_label')
    plt.show()

    continued_network = Net()
    continued_optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    network_state_dict = torch.load('results/task1_model.pth')
    continued_network.load_state_dict(network_state_dict)

    optimizer_state_dict = torch.load('results/task1_optimizer.pth')
    continued_optimizer.load_state_dict(optimizer_state_dict)

    for i in range(4, 9):
        test_counter.append(i * len(train_loader.dataset))
        train(i)
        test()

    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('images/task1_performance_with_8_epochs')
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


if __name__ == '__main__':
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    main()
