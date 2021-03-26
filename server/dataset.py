import numpy as np
from numpy.random import choice
from torchvision.datasets import MNIST
from torchvision.datasets import VisionDataset

class ServerDataset(VisionDataset):
    
    def __init__(self, root, partitions, default_transforms, transform_by_client, seed):
        super().__init__('')
        self.dataset = MNIST(root=root, train=False)
        self.partitions = partitions
        self.default_transforms = default_transforms
        self.transform_by_client = transform_by_client
        
        if seed != None:
            np.random.seed(seed)
    
    def __getitem__(self, index):
        img, target = self.dataset[index]
        trs = choice(self.transform_by_client, 1, p=self.partitions)[0]
        img = trs(self.default_transforms(img))
        return img, target
    
    def __len__(self):
        return len(self.dataset)