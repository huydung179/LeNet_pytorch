import torch

BATCH_SIZE = 32
N_EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAVE_PATH = '/opt/code/pretrained/model.pt'
DATA_ROOT = '/opt/code/data'
print(DEVICE)