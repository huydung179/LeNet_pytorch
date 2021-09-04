import torch
from model import LeNet
from config import *

model = LeNet().to(DEVICE)
model.load_state_dict(torch.load(SAVE_PATH))
example = torch.rand(1, 1, 32, 32).type(torch.FloatTensor).to(DEVICE)
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("/opt/code/pretrained/traced_LeNet_model.pt")
