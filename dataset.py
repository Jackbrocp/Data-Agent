from torch.utils.data import Dataset
import os
import numpy as np
import pickle
import torch
import torchvision
from organize_transform import make_transform, make_magnitude_transform
from torchvision import transforms

DATASET_NUM = {
    'mnist': 60000,
    'cifar10': 50000,
    'cifar100': 50000,
    'tiny-imagenet': 100000,
    'imagenet': 1281167,
}

# class CIFAR10(Dataset):
#     base_folder = ''
#     preix = ''
#     train_list=[[preix+'data_batch_1', 'c99cafc152244af753f735de768cd75f'],
#                 [preix+'data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
#                 [preix+'data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
#                 [preix+'data_batch_4', '634d18415352ddfa80567beed471001a'],
#                 [preix+'data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],]
#     test_list = [['test_batch', '40351d587109b95175f43aff81a1287e'],]
#     meta = {'filename': 'batches.meta',
#             'key': 'label_names',
#             'md5': '5ff9c542aee3614f3951f8cda6e48888',}
#     transform_list = []
#     def __init__(self, root, train=True, transform=None, target_transform=None, aug=None):
#         super(CIFAR10Dataset, self).__init__()
#         self.train = train
#         self.root = root
#         self.data = []
#         self.targets = []
#         self.filename_list = []
#         self.transform = transform
#         self.target_transform = target_transform
#         self.make_magnitude_transform = make_magnitude_transform

#         if self.train:
#             data_list = self.train_list
#         else:
#             data_list = self.test_list
        
#         for file_name, checksum in data_list:
#             file_path = os.path.join(self.root, self.base_folder, file_name)
#             with open(file_path, 'rb') as f:
#                 entry = pickle.load(f, encoding='latin1')
#                 self.data.append(entry['data'])
#                 if self.train:
#                     self.filename_list.extend(entry['filenames'])
#                 if 'labels' in entry:
#                     self.targets.extend(entry['labels'])
#                 else:
#                     self.targets.extend(entry['fine_labels'])

#         self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
#         self.data = self.data.transpose((0, 2, 3, 1))  
#         self._load_meta()
#         self.MAGNITUDE = torch.zeros(50000)
#         self.is_magnitude = False
#         self.full_transform = self.make_magnitude_transform(magnitude=1, cutout_length=16)
#         self.none_transform = self.transform
#         self.warmup_transform = make_transform('cifar10', length=16)[0]

#     def _load_meta(self) -> None:
#         path = os.path.join(self.root, self.base_folder, self.meta['filename'])
#         with open(path, 'rb') as infile:
#             data = pickle.load(infile, encoding='latin1')
#             self.classes = data[self.meta['key']]
#         self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

#     def set_transform(self, magnitude, cutout_length):
#         transform_list = []
#         for i in range(len(magnitude)):
#             transform_list.append(self.make_magnitude_transform(magnitude=magnitude, cutout_length=cutout_length)) 
#         self.transform_list = transform_list
#         return 
    
#     def set_MAGNITUDE(self, idx, magnitude):
#         self.MAGNITUDE[idx] = magnitude

#     def __getitem__(self, index:int):
#         if self.train:
#             img, target = self.data[index], int(self.targets[index]) 
#         else:
#             img, target = self.data[index], int(self.targets[index])
#         if self.target_transform is not None:
#             target = self.target_transform(target)
#         if self.train:
#             if self.is_magnitude:
#                 t = self.make_magnitude_transform(magnitude=self.MAGNITUDE[index].item(), cutout_length=16)
#                 normalized_img = t(img)
#                 full_aug_img = self.full_transform(img)
#                 none_aug_img = self.none_transform(img)
#                 return index, none_aug_img, normalized_img, full_aug_img, target
#             else:   
#                 normalized_img = self.warmup_transform(img)
#             return index, normalized_img, target 
#         else:
#             normalized_img = self.transform(img)
#             return normalized_img, target
        
#     def __len__(self):
#         return len(self.data)
    
#     def extra_repr(self):
#         return "Split: {}".format("Train" if self.train is True else "Test")

class CIFAR10(Dataset):
    def __init__(self, root, train=True, fine_label=True, transform=None, download=False, cls_transform=None, aug=None):
        cifar_dataset = torchvision.datasets.CIFAR10(root=root, train=train, download=download)
        self.data = cifar_dataset.data
        self.labels = cifar_dataset.targets
        
        self.train = train
        self.is_magnitude = False
        # self.make_magnitude_transform = make_magnitude_transform
        self.MAGNITUDE = torch.zeros(len(self.data))
        stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        self.cls_transform = cls_transform  # Store the provided cls_transform
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ]) if cls_transform is None else cls_transform
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        print(self.train_transform)

    def set_MAGNITUDE(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], int(self.labels[index]) 
            normalized_img = self.train_transform(img)
            return index, normalized_img, target
            # if self.is_magnitude:
            #     t = self.make_magnitude_transform(magnitude=self.MAGNITUDE[index].item(), cutout_length=8)
            #     normalized_img = t(img)
            #     return index, normalized_img, target
            # else:
            #     normalized_img = self.warmup_transform(img)
            #     return index, normalized_img, target
        else:
            img, target = self.data[index], int(self.labels[index])
            normalized_img = self.test_transform(img)
            return normalized_img, target
    def __len__(self):
        return len(self.data)

class CIFAR100(Dataset):
    def __init__(self, root, train=True, fine_label=True, transform=None, download=False, cls_transform=None, aug=None):
        cifar_dataset = torchvision.datasets.CIFAR100(root=root, train=train, download=download)
        self.data = cifar_dataset.data
        self.labels = cifar_dataset.targets
        
        self.train = train
        self.is_magnitude = False
        # self.make_magnitude_transform = make_magnitude_transform
        self.MAGNITUDE = torch.zeros(len(self.data))
        stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
        self.cls_transform = cls_transform  # Store the provided cls_transform
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32,padding=4,padding_mode="reflect"),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ]) if cls_transform is None else cls_transform
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])
        print(self.train_transform)

    def set_MAGNITUDE(self, idx, magnitude):
        self.MAGNITUDE[idx] = magnitude
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], int(self.labels[index]) 
            normalized_img = self.train_transform(img)
            return index, normalized_img, target
            # if self.is_magnitude:
            #     t = self.make_magnitude_transform(magnitude=self.MAGNITUDE[index].item(), cutout_length=8)
            #     normalized_img = t(img)
            #     return index, normalized_img, target
            # else:
            #     normalized_img = self.warmup_transform(img)
            #     return index, normalized_img, target
        else:
            img, target = self.data[index], int(self.labels[index])
            normalized_img = self.test_transform(img)
            return normalized_img, target
    def __len__(self):
        return len(self.data)