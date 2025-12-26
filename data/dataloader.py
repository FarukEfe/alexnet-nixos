import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

from model.model import Model

class CustomDataset(Dataset):

    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        """Returns the total number of samples"""
        return len(self.data)
    
    def __getitem__(self, idx):
        """Returns one sample of data at given index"""
        return self.data[idx], self.labels[idx]
    
def load_dummy_data(num_samples=1000, input_shape=(3, 224, 224), num_classes=1000, batch_size=32):
    """Generates dummy data for testing the DataLoader and Model"""
    data = np.random.randn(num_samples, *input_shape)
    labels = np.random.randint(0, num_classes, size=(num_samples,))

    dataset = CustomDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader

# NOTE: THIS ONE WILL BE TESTED, ITS AI GENERATED RN
def load_imagenet_data(data_dir, batch_size=32):
    """Loads ImageNet data from the specified directory using ImageFolder"""
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader