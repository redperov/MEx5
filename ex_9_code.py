import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools


class ConvNet(nn.Module):
    """
    Convolutional neural network.
    """
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2)
        self.mp1 = nn.MaxPool2d(2)
        self.mp2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * 7 * 7, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 10)
        self.bn1 = nn.BatchNorm1d(100)
        self.bn2 = nn.BatchNorm1d(50)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.mp1(x)
        x = F.relu(self.conv2(x))
        x = self.mp2(x)
        x = x.view(-1, 32 * 7 * 7)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.fc3(x)
        return F.softmax(x, dim=1)


def main():

    # Load the data.
    train_loader, test_loader = load_data(False)

    # Split the examples to training and validation sets 80:20.
    train_loader, validation_loader = split_data(train_loader.dataset)

    # Initialize models.
    model_conv = ConvNet()
    model_resnet = create_resnet_model()

    # Initialize hyper-parameters.
    lr = 0.005
    epochs = 10

    # Setting the optimizers.
    optimizer_conv = optim.Adam(model_conv.parameters(), lr=lr)
    optimizer_resnet = optim.SGD(model_resnet.fc.parameters(), lr=0.001, momentum=0.9)
    scheduler = StepLR(optimizer_conv, step_size=3, gamma=0.1)

    # Train and plot the convolution network model.
    train_and_plot(model_conv, optimizer_conv, epochs, train_loader, validation_loader, test_loader, scheduler)

    # Train and plot the ResNet model.
    train_and_plot(model_resnet, optimizer_resnet, epochs, train_loader, validation_loader, test_loader)

    # Write the prediction of the best model to a file.
    write_prediction(model_conv, test_loader)


def load_data(resize_data=False):
    """
    Loads the training and testing data.
    :param resize_data: boolean, sould the data be resized
    :return: train_loader, test_loader
    """
    if resize_data:
        t = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])
    else:
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=True, download=True,
                              transform=t),
        batch_size=64, shuffle=False)

    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, download=True,
                              transform=t),
        batch_size=64, shuffle=False)

    return train_loader, test_loader


def split_data(data_set):
    """
    Splits the training data to training and validation 80:20.
    :param data_set: data_set
    :return: train_loader, validation_loader
    """
    num_train = len(data_set)
    indices = list(range(num_train))
    split = int(num_train * 0.2)

    # Random, non-contiguous split
    validation_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(validation_idx))

    # Perform random split using subset random samples.
    train_sampler = SubsetRandomSampler(train_idx)
    validation_sampler = SubsetRandomSampler(validation_idx)

    train_loader = torch.utils.data.DataLoader(data_set,
                    batch_size=128, sampler=train_sampler)

    validation_loader = torch.utils.data.DataLoader(data_set,
                    batch_size=128, sampler=validation_sampler)

    return train_loader, validation_loader


def create_resnet_model():
    """
    Creates a ResNet model with a custom fully connected layer.
    :return: model
    """
    model_resnet = models.resnet18(pretrained=True)
    for param in model_resnet.parameters():
        param.requires_grad = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_resnet.fc.in_features
    model_resnet.fc = nn.Linear(num_ftrs, 10)

    return model_resnet


def train_and_plot(model, optimizer, epochs, train_loader, validation_loader, test_loader, scheduler=None):
    """
    Trains and plots the results of a given model.
    :param model: model
    :param optimizer: optimizer
    :param epochs: number of epochs
    :param train_loader: train loader
    :param validation_loader: validation loader
    :param test_loader: test loader
    :return: None
    """
    train_loss_list = []
    validation_loss_list = []

    for epoch in range(epochs):

        if scheduler is not None:
            scheduler.step()

        # Train the model.
        train(model, optimizer, train_loader)

        # Get training loss.
        train_loss = test(epoch, model, train_loader, "Training set")
        train_loss_list.append(train_loss)

        # Get validation loss.
        validation_loss = test(epoch, model, validation_loader, "Validation set")
        validation_loss_list.append(validation_loss)

    # Test the model.
    test(0, model, test_loader, "Test set")

    # Plot average training and validation loss VS number of epochs
    plot_graph(epochs, train_loss_list, validation_loss_list)

    # Plot confusion matrix.
    plot_confusion_matrix(model, test_loader)


def plot_graph(epochs, train_loss_list, validation_loss_list):
    """
    Plots a graph of training and validation average loss VS number of epochs.
    :param epochs: number of epochs
    :param train_loss_list: train loss list
    :param validation_loss_list: validation loss list
    :return: None
    """
    epochs_list = list(range(epochs))
    plt.plot(epochs_list, train_loss_list, 'b', label="training loss")
    plt.plot(epochs_list, validation_loss_list, 'r--', label="validation loss")
    plt.xlabel("Epochs")
    plt.ylabel("Average loss")
    plt.legend()
    plt.show()


def plot_confusion_matrix(model, test_loader):
    model.eval()
    test_loss = 0
    y_true = []
    y_pred = []
    for data, target in test_loader:
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        y_true.append(target[0].item())
        y_pred.append(pred[0].item())

    matrix = confusion_matrix(y_true, y_pred)
    classes = [i for i in range(10)]

    plt.imshow(matrix, interpolation='nearest')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # fmt = '.2f' if normalize else 'd'
    thresh = matrix.max() / 2.
    for i, j in itertools.product(range(matrix.shape[0]), range(matrix.shape[1])):
        plt.text(j, i, format(matrix[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if matrix[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def train(model, optimizer, train_loader):
    """
    Trains the network.
    :param model: neural network
    :param optimizer: optimizer
    :param train_loader: train loader
    :return: training average loss
    """
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()
        optimizer.step()


def test(epoch, model, test_loader, set_type):
    """
    Tests the final model.
    :param model: neural network
    :param test_loader: test loader
    :return: test loss
    """
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    for data, target in test_loader:
        output = model(data)
        test_loss += F.cross_entropy(output, target, size_average=False).item()  # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        counter += 1

    test_loss /= len(test_loader.sampler)
    print('\n{}: Epoch: {} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        set_type, epoch, test_loss, correct, len(test_loader.sampler),
        100.0 * correct / len(test_loader.sampler)))

    return test_loss


def write_prediction(model, test_loader):
    """
    Performs a prediction over the test set and writes it to a file.
    :param model: model
    :param test_loader: test loader
    :return: None
    """
    with open("test.pred", "w") as test_file:
        for data, target in test_loader:

            # Pass the example through the classifier.
            output = model(data)

            # Extract the predicted label.
            predicted_value = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability

            for value in predicted_value:
                test_file.write(str(value.item()) + '\n')


if __name__ == "__main__":
    main()
