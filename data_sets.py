from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from config import DATASET_NAME

def load_cifar10_test(batch_size=1, shuffle=True):
    test_data_set = datasets.CIFAR10(root="./data/cifar10", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)
    return test_loader

def load_cifar100_test(batch_size=1, shuffle=True):
    test_data_set = datasets.CIFAR100(root="./data/cifar100", train=False, download=True, transform=transforms.ToTensor())
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)
    return test_loader

def load_imagenet_test(batch_size=1, shuffle=True):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    test_data_set = datasets.ImageFolder(root="./data/imagenet", transform=transform)
    test_loader = DataLoader(test_data_set, batch_size=batch_size, shuffle=shuffle)
    return test_loader

def load_test_data(dataset_name="mnist", batch_size=1, shuffle=True, attack_oriented=True):
    """
    Load test data for the specified dataset.

    Args:
        dataset_name (str): Name of the dataset to load ("mnist", "cifar10", "cifar100" or "imagenet").
        batch_size (int): Batch size for the UnifiedDataSetLoader. Batch size must be 1 for an adversarial attack.
        shuffle (bool): Whether to shuffle the data.
        attack_oriented (bool): Whether the test data is for adversarial attack or not. If attack_oriented is True, the batch size must be 1.
    
    Returns:
        test_data_loader: A DataLoader to load the target test set in batch.
    """
    assert (not attack_oriented) or (batch_size == 1), "Batch size must be 1 for an adversarial attack!"
    assert dataset_name in DATASET_NAME, f"Dataset {dataset_name} is not supported!"
    if dataset_name == "cifar10":
        return load_cifar10_test(batch_size=batch_size, shuffle=shuffle)
    elif dataset_name == "cifar100":
        return load_cifar100_test(batch_size=batch_size, shuffle=shuffle)
    elif dataset_name == "imagenet":
        return load_imagenet_test(batch_size=batch_size, shuffle=shuffle)