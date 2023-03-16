import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from task1 import Net


class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0, 0), 1/120, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)


def main():
    network = Net()
    network.conv2_drop = nn.Dropout2d(p=0.2)

    network.load_state_dict(torch.load('results/task1_model.pth'))
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    for param in network.parameters():
        param.requires_grad = False

    greek_train = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./greek_train_extra', transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), GreekTransform(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size_train, shuffle=True
    )

    greek_test = torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder('./greek_test_extra', transform=torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor(), GreekTransform(),
             torchvision.transforms.Normalize((0.1307,), (0.3081,))])), batch_size=batch_size_test, shuffle=True
    )

    network.fc2 = nn.Linear(50, 24)
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    print(network)

    examples = enumerate(greek_train)
    batch_idx, (example_data, example_target) = next(examples)

    print(example_data.shape)

    plt.figure()
    for i in range(batch_size_train):
        plt.subplot(1, batch_size_train, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'images/task4_greek_letter_outputs')
    plt.show()

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i * len(greek_train.dataset) for i in range(n_epochs + 1)]

    def train(epoch):
        network.train()
        for batch_idx, (data, target) in enumerate(greek_train):
            optimizer.zero_grad()
            output = network(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print(
                    f'Train epoch: {epoch} [{batch_idx * len(data)}/{len(greek_train.dataset)} ({100. * batch_idx / len(greek_train):.0f}%)]\tLoss: {loss.item():.6f}')
                train_losses.append(loss.item())
                train_counter.append(batch_idx * 64 + (epoch - 1) * len(greek_train.dataset))
                torch.save(network.state_dict(), 'results/extension2_model.pth')
                torch.save(optimizer.state_dict(), 'results/extension2_optimizer.pth')

    def test():
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in greek_test:
                output = network(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(greek_test.dataset)
        test_losses.append(test_loss)
        print(
            f'Test set: Avg. loss: {test_loss:.4f}, Accuracy: {correct}/{len(greek_test.dataset)} ({100. * correct / len(greek_test.dataset):.0f}%)\n')

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    plt.savefig('images/extension2_greek_letter_performance')
    plt.show()

    with torch.no_grad():
        output = network(example_data)
    plt.figure()
    for i in range(batch_size_train):
        plt.subplot(1, batch_size_train, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title(f'Prediction: {output.data.max(1, keepdim=True)[1][i].item()}')
        plt.xticks([])
        plt.yticks([])
    plt.savefig('images/extension2_greek_letter_outputs_with_prediction_label')
    plt.show()


if __name__ == '__main__':
    n_epochs = 10
    learning_rate = 0.5
    batch_size_train = 5
    batch_size_test = 9
    momentum = 0.5
    log_interval = 10
    random_seed = 1
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    main()
