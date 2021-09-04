from torchvision.transforms import Compose, ToTensor, Resize, Normalize

data_transform = Compose([
    ToTensor(),
    # Normalize(mean=[0.1307], std=[0.3081]),
    Resize(32)
])