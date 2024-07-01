import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision import datasets
from torchvision.transforms import ToTensor

def get_fashion_mnist_datasets(root="data"):
    training_data = datasets.FashionMNIST(
        root=root,
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.FashionMNIST(
        root=root,
        train=False,
        download=True,
        transform=ToTensor()
    )
    
    return training_data, test_data

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def get_dataloaders(training_data, test_data, batch_size=64):
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return train_dataloader, test_dataloader
