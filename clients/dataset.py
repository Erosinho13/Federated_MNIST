import random
import numpy as np
from torch.utils.data import Subset
from torchvision.datasets import MNIST
from torchvision.datasets import VisionDataset


class ClientDataset(VisionDataset):
    
    def __init__(self, data, transform=None):
        super().__init__('', transform=transform)
        self.transform = transform
        self.data = data
        
    def __getitem__(self, index):
        if self.transform is not None:
            return self.transform(self.data[index][0]), self.data[index][1]
        return self.data[index][0], self.data[index][1]
        
    def __len__(self):
        return len(self.data)


class Federator:
    
    def __init__(self, root, default_transforms=None, 
                 partitions=np.array([.5,.5]), transform_by_client=None, seed=None):
        
        self.dataset = MNIST(root=root, transform=default_transforms)
        self.K = len(partitions)
        self.transform_by_client = transform_by_client
        
        if seed != None:
            random.seed(seed)
        
        if round(sum(partitions),2) != 1:
            raise Exception("Partitions array does not sum to 1")
        if transform_by_client != None:
            if len(self.transform_by_client) != len(partitions):
                raise Exception("Lengths of transform_by_client and partitions must correspond")
        
        self.clients_idxs = self.__partition(partitions, len(self.dataset))
        self.client_datasets = [self.__generateClientDataset(k) for k in range(self.K)]
        
    def __partition(self, p, n):
    
        p = (n*p).astype(int)
        l = list(range(sum(p)))
        random.shuffle(l)

        return [l[sum(p[:i]):sum(p[:i])+p[i]] for i in range(len(p))]
        
    def __generateClientDataset(self, k):
        if self.transform_by_client is not None:
            dataset = ClientDataset(Subset(self.dataset, self.clients_idxs[k]),
                                    transform=self.transform_by_client[k])
        else:
            dataset = ClientDataset(Subset(self.dataset, self.clients_idxs[k]))
        return dataset