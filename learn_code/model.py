import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 100, 100)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 100, 100)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 50, 50)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 50, 50)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 50, 50)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 25, 25)
        )
        self.out = nn.Linear(32 * 25 * 25, 4)   # fully connected layer, output 4 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x                    # return x for visualization


class AngleCNN(nn.Module):
    def __init__(self):
        super(AngleCNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 100, 100)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 100, 100)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 50, 50)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 50, 50)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 50, 50)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 25, 25)
        )
        self.out = nn.Linear(32 * 25 * 25, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x                    # return x for visualization


class AngleCNN1(nn.Module):
    def __init__(self):
        super(AngleCNN1, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 100, 100)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=5,              # filter size
                stride=1,                   # filter movement/step
                padding=2,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 100, 100)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 50, 50)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 50, 50)
            nn.Conv2d(16, 32, 5, 1, 2),     # output shape (32, 50, 50)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 25, 25)
        )
        self.fc0 = nn.Linear(32 * 25 * 25, 16 * 5 * 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = self.fc0(x)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.out(x)
        return output, x                    # return x for visualization


class AngleCNN_2(nn.Module):
    def __init__(self):
        super(AngleCNN_2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


