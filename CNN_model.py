import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=(5, 5), stride=(1, 1), padding=2)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        self.dense1 = nn.Linear(57600, 512)
        self.dense2 = nn.Linear(512, output_channels)

        nn.init.xavier_uniform_(self.conv1.weight, gain=1)
        self.conv1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1)
        self.conv2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.conv3.weight, gain=1)
        self.conv3.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dense1.weight, gain=1)
        self.dense1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dense2.weight, gain=1)
        self.dense2.bias.data.fill_(0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)
        return x
