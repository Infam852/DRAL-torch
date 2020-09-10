import os

from tqdm import tqdm

import torch
import torch.optim as optim
import torch.nn as nn

from models import ConvNet
from data_manipulation.loader import DataLoader
from dral.config import CONFIG
from utils import show_img, show_grid_imgs


""" Train convnet and then save the model """
DATASETS_DICT = './data'
IMG_SIZE = CONFIG['img_size']

# x_train = DataLoader.load(os.path.join(DATASETS_DICT, 'x_train_cats_dogs.npy'))
# y_train = DataLoader.load(os.path.join(DATASETS_DICT, 'y_train_cats_dogs.npy'))
# x_train = DataLoader.load(os.path.join(DATASETS_DICT, 'x_cats_dogs_skimage.npy'))
# y_train = DataLoader.load(os.path.join(DATASETS_DICT, 'y_cats_dogs_skimage.npy'))

x_train = DataLoader.load(os.path.join(DATASETS_DICT, 'x_rps_skimage.npy'))
y_train = DataLoader.load(os.path.join(DATASETS_DICT, 'y_rps_skimage.npy'))

x_train = torch.Tensor(x_train).view(-1, IMG_SIZE, IMG_SIZE)
y_train = torch.Tensor(y_train)
print(x_train.shape)
print(y_train.shape)

N_TRAIN = CONFIG['n_train']
N_EVAL = CONFIG['n_eval']
N_TEST = CONFIG['n_test']

if N_TRAIN + N_EVAL + N_TEST > len(x_train):
    raise Exception('Not enough data!')


# resnet50 works with 224, 244 input size
net = ConvNet()
optimizer = optim.Adam(net.parameters(), lr=1e-3)
loss_function = nn.MSELoss()

# split data
x_eval = x_train[:N_EVAL]
y_eval = y_train[:N_EVAL]

x_test = x_train[N_EVAL:N_EVAL+N_TEST]
y_test = y_train[N_EVAL:N_EVAL+N_TEST]

x_train = x_train[N_EVAL+N_TEST:N_EVAL+N_TEST+N_TRAIN]
y_oracle = y_train[N_EVAL+N_TEST:N_EVAL+N_TEST+N_TRAIN]

show_grid_imgs(x_train[:9], y_oracle[:9], (3, 3))

EPOCHS = 8
BATCH_SIZE = 32

print('Start training')
for epoch in range(EPOCHS):
    for k in tqdm(range(0, len(x_train), BATCH_SIZE)):
        batch_x = x_train[k:k+BATCH_SIZE].view(-1, 1, IMG_SIZE, IMG_SIZE)
        batch_y = y_oracle[k:k+BATCH_SIZE]

        net.zero_grad()

        out = net(batch_x)
        loss = loss_function(out, batch_y)
        loss.backward()
        optimizer.step()

    print(f"Epoch: {epoch}. Loss: {loss}")

correct = 0
total = 0

with torch.no_grad():
    for k in tqdm(range(len(x_test))):
        real_class = torch.argmax(y_test[k])
        net_out = net(x_test[k].view(-1, 1, IMG_SIZE, IMG_SIZE))[0]  # returns list
        predicted_class = torch.argmax(net_out)

        if predicted_class == real_class:
            correct += 1
        total += 1

print("Accuracy: ", round(correct/total, 3))

torch.save(net, f'{DATASETS_DICT}/cnn_rps_model.pt')
