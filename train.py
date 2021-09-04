import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data_augmentation import data_transform
from model import LeNet
from config import *

def train_1_epoch(train_loader, model, loss, optim, device):
    model.train()
    train_loss = []
    corrects = 0
    n_train = 0
    training_loop = tqdm(train_loader, leave=False, ncols=100)
    for x, y in training_loop:
        x = x.to(device)
        y = y.to(device)
        pred = model(x)

        batch_loss = loss(pred, y)
        train_loss.append(batch_loss.item())
        batch_loss.backward()

        optim.step()
        optim.zero_grad()

        pred = pred.argmax(dim=1)
        corrects += (pred == y).sum().float()
        n_train += y.shape[0]

        training_loop.set_postfix(loss=sum(train_loss) / len(train_loss))
    accuracy = corrects / n_train
    return sum(train_loss) / len(train_loss), accuracy


def validation(val_loader, model, loss, device):
    val_loss = []
    corrects = 0
    n_val = 0
    with torch.no_grad():
        model.eval()
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)

            batch_loss = loss(pred, y)
            val_loss.append(batch_loss.item())
            pred = pred.argmax(dim=1)
            corrects += (pred == y).sum().float()
            n_val += y.shape[0]
        
        accuracy = corrects / n_val
        val_loss = sum(val_loss) / len(val_loss)
    return val_loss, accuracy


def train(train_loader, val_loader, model, loss, optim, n_epochs, device, save_path, early_stop=5):    
    best_loss = 999.
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}: ")
        train_loss, train_acc = train_1_epoch(train_loader, model, loss, optim, device)
        cur_loss, cur_acc = validation(val_loader, model, loss, device)
        print(f"Loss in training set: {train_loss:.02f}, Acc: {train_acc:.02f}. Loss in validation set: {cur_loss:.02f}, Acc: {cur_acc:.02f}")
        if cur_loss < best_loss:
            n = 0
            best_loss = cur_loss
            print(f"Loss decreased to {cur_loss:.02f}, model saved to {save_path}")
            torch.save(model.state_dict(), save_path)
        else:
            n += 1
        
        if n >= early_stop:
            print(f"No improvement in validation set in the recent {early_stop} epochs. Stopped")


def main():
    DATA_ROOT = '/opt/code/data'
    dataset = MNIST(DATA_ROOT, train=True, download=True, transform=data_transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
    model = LeNet().to(DEVICE)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    train(train_loader, val_loader, model, loss, optim, N_EPOCHS, DEVICE, SAVE_PATH)


if __name__ == '__main__':
    main()