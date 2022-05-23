# This file is the dataloader file for pytorch 1.0.1 library
# This dataset is generated by Yi-Chun and Boyi in Jan 2019
# You have any problem, you could contact me through email:nick831111@gmail.com

import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.transform import radon
from torch import from_numpy
from torch.utils.data import Dataset
from torch.distributions.normal import Normal
from PIL import Image
from os import listdir
from natsort import natsort
from os.path import join, splitext
from retreive_noise import fit_noise
from tqdm import tqdm
from natsort import natsorted
import pdb
from torch.utils.data import DataLoader

class THz_dataloader(Dataset):
    def __init__(self, root_dir, data_dir_list, target_dir_list, offset, noise_file, pad_num, if_sift):
        # self.data_path and self.target_path is a pair of data and target going to feed into the model.
        self.data_path = []
        self.target_path = []
        self.noise = np.load(noise_file)
        self.mu, self.sigma = fit_noise(self.noise)
        self.pad_num = pad_num
        self._if_sift = if_sift

        for data_dir, target_dir in zip(data_dir_list,target_dir_list):
            obj_name = target_dir.split("_")[2]
            idx_offset = offset.image_offset[obj_name]
            target_names = [i for i in listdir(join(root_dir,target_dir)) if i.endswith('.png')]
            target_names = natsorted(target_names)
            for target_f in target_names:
                num = int(splitext(target_f)[0])
                data_f = obj_name + '_' + str(num+idx_offset) + ".npy"
                if num < offset.upper_offset[obj_name] and num > offset.lower_offset[obj_name]:
                    self.data_path.append(join(root_dir,data_dir,data_f))
                    self.target_path.append(join(root_dir,target_dir,target_f))


    def _pad_noise(self, data, mu, sigma, pad_num):
        m = Normal(mu, sigma)
        noise = m.sample((data.shape[1] * data.shape[2] * pad_num, ))
        noise = noise.view(data.shape[1], data.shape[2], pad_num)
        idx = np.random.randint(0, pad_num, size = 1)[0]
        return torch.cat((noise[None, :, :, :idx], data, noise[None, :, :, idx:]), dim=-1)

    def crop2center(self, data, end):
        idx = 3
        return data[:-1,idx:end-idx,:]

    def crop2center_size(self, data, size):
        height = data.shape[1]
        leftover = (height - size) // 2

        center_data = data[:,leftover:height-leftover,:]
        return center_data

    def normalize(self, data):
        data = data - torch.min(data)
        data = data / torch.max(torch.abs(data))
        data = data - 0.5
        return data

    def __len__(self):
        return len(self.target_path)

    def __getitem__(self, idx):
        try:
            data = np.load(self.data_path[idx]).astype(np.float32) # angle x width x t
        except:
            print(idx)
            print(self.data_path[idx])
            print(np.load(self.data_path[idx]).shape)
            print(data.shape)
        data = self.crop2center_size(data, 286)

        data = from_numpy(data).permute(0,1,2)[None,:,:,:] #to tensor Channel x Angle x Width(Sample range) x D(time domain)
        if self.pad_num != 0:
            data = self._pad_noise(data, self.mu, self.sigma, self.pad_num)
        data = self.normalize(data)

        fn = lambda x : 1 if x<255 else 0
        target = Image.open(self.target_path[idx]).convert("L").point(fn,mode='1') #Convert Image to binary mask
        target = np.array(target)
        target = target[1:-1,:]
        target = radon(target,theta=np.arange(60)*6,circle=True).astype(np.float32)
        target = from_numpy(target.T) # to tensor : Angle x Width(Sample range)
        target = -1 * target

        if self._if_sift:
            angle_idx = np.random.choice(60, 20, replace=False)
        else:
            angle_idx = np.arange(30)

        return data[:, angle_idx, :, :], target[angle_idx, :], self.data_path[idx], self.target_path[idx] #, self.data_path[idx]


