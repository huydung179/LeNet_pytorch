import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data_augmentation import data_transform
from model import LeNet
from config import *


def validation(val_loader, model, device):
    corrects = 0
    n_val = 0
    with torch.no_grad():
        model.eval()
        for x, y in tqdm(val_loader, leave=False):
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            pred = pred.argmax(dim=1)
            corrects += (pred == y).sum().float()
            n_val += y.shape[0]
        
        accuracy = corrects / n_val
    return accuracy


def main():
    dataset = MNIST(DATA_ROOT, train=False, download=True, transform=data_transform)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    model = LeNet().to(DEVICE)
    model.load_state_dict(torch.load(SAVE_PATH))
    acc = validation(dataloader, model, DEVICE)
    print(f'Accuracy on test set: {acc * 100:.02f}%')


if __name__ == '__main__':
    main()