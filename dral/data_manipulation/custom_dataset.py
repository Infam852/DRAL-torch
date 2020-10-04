import torch
from torch.utils.data import Dataset, DataLoader
from dral.config.config_manager import ConfigManager
from dral.utils import LOG
import pandas as pd
import os
import cv2
from tqdm import tqdm
from torchvision import transforms
import PIL

import torch.optim as optim
import torch.nn as nn

from dral.models import ConvNet
import numpy as np
import torchvision


def remove_corrupted_images(path):
    removed_files = []
    for f in os.listdir(path):
        full_path = os.path.join(path, f)
        try:
            img = cv2.imread(full_path)
            if img is None:
                raise OSError
        except OSError:
            os.remove(full_path)
            removed_files.append(full_path)
    LOG.info(f'Removed files: {removed_files}')


def create_csv_file(target_file, data_dir, skips=None):
    if skips is None:
        skips = []

    with open(target_file, 'w+') as tf:
        label = 0
        for fdir in sorted(os.listdir(data_dir)):
            if fdir in skips:
                continue

            dirpath = os.path.join(data_dir, fdir)
            if os.path.isdir(dirpath):
                LOG.info(f'Start loading from {dirpath}...')
                for f in tqdm(os.listdir(dirpath)):
                    feature_rpath = os.path.join(fdir, f)
                    tf.write(f'{feature_rpath},{label}\n')
                label += 1
    LOG.info(f'CSV file {target_file} saved')


class TrainDataset(Dataset):

    def __init__(self, csv_path, root_dir, transforms=None):
        self.root_dir = root_dir
        self.annotations = pd.read_csv(csv_path)
        self.transforms = transforms
        self.n_labels = 2  # !TODO

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[idx, 0])
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)

        label = torch.tensor(int(self.annotations.iloc[idx, 1]))
        label = np.eye(self.n_labels)[-label-1]
        if self.transforms:
            img = self.transforms(img)

        return img, label


if __name__ == '__main__':
    root_dir = './data/TestImages'
    csv_file = './data/annotations.csv'
    # remove_corrupted_images(root_dir + '/Dog')
    # remove_corrupted_images(root_dir + '/Cat')
    # create_csv_file(csv_file, root_dir,
    #                 skips=['Unknown'])

    final_img_size = 128
    transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.ColorJitter(),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(final_img_size),
        transforms.ToTensor()
    ])
    td = TrainDataset(csv_file, root_dir, transforms=transforms)
    dataloader = DataLoader(td, batch_size=128,
                            shuffle=True, num_workers=0)

    model = torchvision.models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, 500),
        nn.Linear(500, 2)
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[500,1000,1500], gamma=0.5)
    epochs = 3
    itr = 1
    p_itr = 200
    model.train()
    total_loss = 0
    loss_list = []
    acc_list = []
    for epoch in range(epochs):
        for samples, labels in dataloader:
            optimizer.zero_grad()
            output = model(samples)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            scheduler.step()
            
            if itr%p_itr == 0:
                pred = torch.argmax(output, dim=1)
                correct = pred.eq(labels)
                acc = torch.mean(correct.float())
                print('[Epoch {}/{}] Iteration {} -> Train Loss: {:.4f}, Accuracy: {:.3f}'.format(epoch+1, epochs, itr, total_loss/p_itr, acc))
                loss_list.append(total_loss/p_itr)
                acc_list.append(acc)
                total_loss = 0
                
            itr += 1
    # itr = iter(dataloader)
    # x, y = next(itr)

    # n_output = 2
    # net = ConvNet(n_output, final_img_size)
    # optimizer = optim.Adam(net.parameters(), lr=1e-3)
    # loss_function = nn.MSELoss()

    # EPOCHS = 10

    # print('Start training')
    # for epoch in range(EPOCHS):
    #     for batch_x, batch_y in tqdm(dataloader):

    #         net.zero_grad()

    #         out = net(batch_x).double()
    #         loss = loss_function(out, batch_y)
    #         loss.backward()
    #         optimizer.step()

    #     print(f'Epoch: {epoch}. Loss: {loss}')

    # correct = 0
    # total = 0
