from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from dral.config import CONFIG
from dral.data_manipulation.dataset_manager import DatasetsManager
from dral.data_manipulation.loader import DataLoader
from dral.config.config_manager import ConfigManager


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
    def __init__(self, n_output, img_size):
        super().__init__()
        self.img_size = img_size
        if self.img_size % (2**3) > 0:
            raise ValueError('Wrong image size!')

        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
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


if __name__ == '__main__':
    cm = ConfigManager('testset')
    print(cm.get_dataset_path())
    imgs = DataLoader.get_images_objects(
        cm.get_dataset_path(), 'processed_x.npy',
        'processed_y.npy', to_tensor=True)
    dm = DatasetsManager(cm, imgs)
    print(dm)
    model = torchvision.models.densenet121(pretrained=True)

    num_ftrs = model.classifier.in_features
    print(num_ftrs)
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500, 1000, 1500], gamma=0.5)

    epochs = 3
    itr = 1
    batch_size = 64
    p_itr = 200
    model.train()
    total_loss = 0
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        for start_idx in range(0, len(dm.train), batch_size):
            samples = dm.train.get_x(
                list(range(start_idx, start_idx+batch_size)))
            labels = torch.Tensor(dm.train.get_y(
                list(range(start_idx, start_idx+batch_size))))
            print(samples.shape)
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()

            if itr % p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                acc = torch.mean(correct.float())
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
                loss_list.append(total_loss/p_itr)
                acc_list.append(acc)
                total_loss = 0
                
        itr += 1