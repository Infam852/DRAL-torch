from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from dral.config import CONFIG


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return F.log_softmax(x, dim=1)


class ConvNet(nn.Module):
    """
    Output layer size: W_out = (W_in - kernel + 2*padding) / stride + 1
    """
    def __init__(self, n_output):
        super().__init__()
        self.img_size = CONFIG['img_size']
        if self.img_size % (2**3) > 0:
            raise ValueError('Wrong image size!')

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        self.dropout2 = nn.Dropout(0.2)
        self.dropout3 = nn.Dropout(0.3)
        # input: 32x32 -> 8x8xNchannels (2 pooling)
        # 3 - number of pooling layers (assumption: conv layers do not change w, h)
        dim = (self.img_size // (2**3))
        self.fc_input_size = dim * dim * 128
        self.fc1 = nn.Linear(self.fc_input_size, 256)
        self.fc2 = nn.Linear(256, n_output)

        self.optimizer = optim.Adam(self.parameters(), lr=0.001)
        self.loss_function = nn.MSELoss()

    def forward(self, x):
        x = self.dropout2(self.pool(F.relu(self.conv1(x))))
        x = self.dropout2(self.pool(F.relu(self.conv2(x))))
        x = self.dropout2(self.pool(F.relu(self.conv3(x))))
        x = x.view(-1,  self.fc_input_size)  # ! change img size change NxNx
        x = self.dropout2(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def evaluate(self, x, y):
        correct = 0
        total = 0
        with torch.no_grad():
            for k in tqdm(range(len(x))):
                real_class = torch.argmax(y[k])
                net_out = self.forward(self._correct_view(x[k]))[0]
                predicted_class = torch.argmax(net_out)

                if predicted_class == real_class:
                    correct += 1
                total += 1
        print('Accuracy: ', round(correct/total, 3))

    def fit(self, x, y, epochs, batch_size=8):
        x = torch.Tensor(x.float())
        x = self._correct_view(x)
        y = torch.Tensor(y.float())
        for epoch in range(epochs):
            for k in tqdm(range(0, len(x), batch_size)):
                batch_X = self._correct_view(x[k:k+batch_size])
                batch_y = y[k:k+batch_size]

                self.zero_grad()

                outputs = self(batch_X)
                loss = self.loss_function(outputs, batch_y)
                loss.backward()
                self.optimizer.step()    # Does the update

            print(f'Epoch: {epoch}. Loss: {loss}')

    def _correct_view(self, x):
        return x.view(-1, 1, self.img_size, self.img_size)
