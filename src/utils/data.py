import numpy as np
import os
import pickle

from typing import List
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader, Subset, sampler


def deep_dataloader(data, train, batch_size=256, data_size=None, shuffle=False, save_dir=None, seed=0, augment=False):
    """
    Creates a dataloader with mini-batches (X, y) where X are normalized images [0, 1] and y the corresponding labels.
    If shuffle is true, the training data is shuffled at every iteration. The test data is only shuffled once
    when the dataloader is initialized in order to enable easy comparison of test images across algorithms.

    Parameters
    ----------
    data : dataset
    train : True for training set, False for test set
    batch_size : number of samples in each batch. -1 enables the full-batch setting (not recommended for large datasets)
    data_size : target length of the dataset. If None, the current length is used
    shuffle : whether to shuffle the data
    save_dir : directory of data folder where the datasets are stored
    seed  : seed for controlling the sequence of sampling and shuffling
    augment : whether to augment the dataset with randomly cropped and horizonally flipped images

    Returns
    -------
    data_loader : configured dataloader object
    """

    data_path = os.path.join(save_dir, 'data') if save_dir else 'data'
    t_list = [transforms.ToTensor()]  # normalizes to [0, 1]

    # dataset menu
    if data == 'fashion_mnist':
        if augment:
            t_list.extend([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip()])
        dataset = datasets.FashionMNIST(root=data_path,
                                        train=train,
                                        download=True,
                                        transform=transforms.Compose(t_list))
    elif data == 'cifar_10':
        if augment:
            t_list.extend([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()])
        dataset = datasets.CIFAR10(root=data_path,
                                   train=train,
                                   download=True,
                                   transform=transforms.Compose(t_list))
    elif data == 'celeba':
        if augment:
            raise NotImplementedError('augmentation not supported for celeba')
        dataset = get_celeba_dataset(train=train,
                                     data_path=data_path)
    # full batch setting
    if batch_size == -1:
        batch_size = data_size or len(dataset)

    # get data-loader
    data_loader = get_dataloader(dataset=dataset,
                                 train=train,
                                 batch_size=batch_size,
                                 shuffle=shuffle,
                                 seed=seed,
                                 data_size=data_size,
                                 num_workers=5)
    return data_loader


def data_sampler(data, data_len=None, random=True, seed=0):
    """
    Creates a data sampler that can consist of a subset of the data.

    Parameters
    ----------
    data : dataset
    data_len : target length of the dataset. If None, the current length is used
    random : whether a random sampler is required; creates new batches at every iteration based on shuffled data
    seed  : seed for controlling the sequence of sampling and shuffling

    Returns
    -------
    sampler : data sampler
    """
    if not data_len:
        data_len = len(data)

    idx = list(range(data_len))

    np.random.seed(seed)
    np.random.shuffle(idx)
    return sampler.SubsetRandomSampler(idx) if random else Subset(data, idx)


def get_dataloader(dataset, train, batch_size, shuffle=False, seed=0, data_size=None, num_workers=0):
    """"
    Creates and configures a dataloader object. It represents a Python iterable over the dataset.

    Parameters
    ----------
    dataset : pytorch dataset
    train : True for training set, False for test set
    batch_size : number of samples in each batch. -1 enables the full-batch setting (not recommended for large datasets)
    shuffle : whether to shuffle the data
    seed  : seed for controlling the sequence of sampling and shuffling
    data_size : target length of the dataset. If None, the current length is used
    num_workers : a value > 0 turns on multi-process data loading with the specified number of loader worker processes

    Returns
    -------
    data_loader : configured dataloader object
    """
    if train:
        if data_size:  # sample a data subset. sampler returns randomly sampled images in batches
            sampler = data_sampler(dataset, data_size)
            data_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     sampler=sampler,
                                     num_workers=num_workers)
        else:
            data_loader = DataLoader(dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle,
                                     num_workers=num_workers)
    else:
        if shuffle:   # shuffle only at run-time i.e. for every seed
            dataset = data_sampler(dataset,
                                   data_size,
                                   random=False,
                                   seed=seed)
        data_loader = DataLoader(dataset,
                                 batch_size=batch_size,
                                 num_workers=num_workers)
    return data_loader


def get_celeba_dataset(train, data_path):
    """"
    Original CelebA data downloaded from : https://www.kaggle.com/jessicali9530/celeba-dataset
    We store indices of train, and test images in subsample_train_indices.pickle and
    subsample_test_indices.pickle respectively.

    Parameters
    ----------
    train : True for training set, False for test set
    data_path : path of the data; equals './data" by default

    Returns
    -------
    dataset : pytorch dataset with celeba images
    """

    class CustomCelebADataset(Dataset):
        def __init__(self, dataset_subset: Dataset, class_labels: List[int]):
            assert len(dataset_subset) == len(class_labels)
            self.dataset_subset = dataset_subset
            self.class_labels = class_labels

        def __len__(self):
            return len(self.class_labels)

        def __getitem__(self, index):
            # Dataset subset has the wrong class labels
            # __getitem__ here is returning a tuple of x, y.
            # Overwrite y with our own custom record of class labels
            Xy = self.dataset_subset[index]
            return Xy[0], self.class_labels[index]

    t_list = [transforms.ToTensor(), transforms.Resize((64, 48))]

    img_path = data_path + '/celeba/img_align_celeba'
    full_dataset = datasets.ImageFolder(root=img_path, transform=transforms.Compose(t_list))

    def _load_data_from_path(full_path: str) -> CustomCelebADataset:
        with open(full_path, 'rb') as f:
            data_dict = pickle.load(f)
        subset_dataset = Subset(full_dataset, data_dict['idx'])
        return CustomCelebADataset(subset_dataset, data_dict['y'])

    dataset = _load_data_from_path(data_path + '/celeba/subsample_' + ('train' if train else 'test')
                                   + '_indices.pickle')
    return dataset

