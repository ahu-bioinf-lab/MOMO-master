import pytorch_lightning as pl
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import subprocess
import pickle
import selfies as sf
import csv
from tqdm import tqdm
import os
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
import time
import math


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

delta_g_to_kd = lambda x: math.exp(x / (0.00198720425864083 * 298.15))


class MolDataModule(pl.LightningDataModule):
    def __init__(self, batch_size, file):
        super(MolDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = Dataset(file)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.8)), int(round(len(self.dataset) * 0.2))])
    
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=16, pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True, num_workers=16, pin_memory=True)
    
    
class PropDataModule(pl.LightningDataModule):
    def __init__(self, x, y, batch_size):
        super(PropDataModule, self).__init__()
        self.batch_size = batch_size
        self.dataset = TensorDataset(x, y)
        self.train_data, self.test_data = random_split(self.dataset, [int(round(len(self.dataset) * 0.9)), int(round(len(self.dataset) * 0.1))])
        
    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.batch_size, drop_last=True)



if os.path.exists('dm.pkl'):
    dm = pickle.load(open('dm.pkl', 'rb'))