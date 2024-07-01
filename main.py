import torch
from datasets import get_fashion_mnist_datasets, get_dataloaders
from utils import labels_map, plot_sample_images, show_batch_sample

def main():
    training_data, test_data = get_fashion_mnist_datasets()
    
    print("Training data size:", len(training_data))
    print("Test data size:", len(test_data))

    plot_sample_images(training_data, labels_map)
    
    train_dataloader, test_dataloader = get_dataloaders(training_data, test_data)
    
    show_batch_sample(train_dataloader)

if __name__ == "__main__":
    main()
