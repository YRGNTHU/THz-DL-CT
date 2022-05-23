import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.transform import radon

class THz_SR_net(nn.Module):
    def __init__(self, sp_k_size=3, t_k_size=3, out_ch_tconv1=64):
        super(THz_SR_net, self).__init__()
        self.out_ch_tconv1 = out_ch_tconv1
        self.sp_k_s = sp_k_size
        self.t_k_s = t_k_size


        # Block 1 cbr-cbr-maxpool
        block1 = nn.Sequential(
            nn.Conv2d(1, self.out_ch_tconv1, kernel_size=(1,self.t_k_s),
                            stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv1),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv1, self.out_ch_tconv1, kernel_size=(1,self.t_k_s),
                            stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv1),
            nn.ReLU(),
            nn.MaxPool2d((1,2),stride=(1,2))
            )
        self.out_ch_tconv4 = self.out_ch_tconv1 * 2

        # Block 2 cbr-cbr-maxpool
        block2 = nn.Sequential(
            nn.Conv2d(self.out_ch_tconv1,self.out_ch_tconv4,kernel_size=(1,self.t_k_s),
                            stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv4),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv4,self.out_ch_tconv4,kernel_size=(1,self.t_k_s),
                            stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv4),
            nn.ReLU(),
            nn.MaxPool2d((1,2),stride=(1,2))
            )

        self.out_ch_tconv7 = self.out_ch_tconv4 * 2

        # Block 3 cbr-cbr-cbr-maxpool
        block3 = nn.Sequential(
            nn.Conv2d(self.out_ch_tconv4,self.out_ch_tconv7,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv7),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv7,self.out_ch_tconv7,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv7),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv7,self.out_ch_tconv7,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv7),
            nn.ReLU(),
            nn.MaxPool2d((1,2),stride=(1,2))
            )

        self.out_ch_tconv11 = self.out_ch_tconv7 * 2

        # Block 4 cbr-cbr-cbr-maxpool
        block4 = nn.Sequential(
            nn.Conv2d(self.out_ch_tconv7,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.MaxPool2d((1,2),stride=(1,2))
            )

        # Block 5 cbr-cbr-cbr-maxpool
        block5 = nn.Sequential(
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.MaxPool2d((1,2),stride=(1,2))
            )
        block6 = nn.Sequential(
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.Conv2d(self.out_ch_tconv11,self.out_ch_tconv11,kernel_size=(1,self.t_k_s),
                        stride=(1,1), padding=(0,int(self.t_k_s//2))),
            nn.BatchNorm2d(self.out_ch_tconv11),
            nn.ReLU(),
            nn.MaxPool2d((1,2),stride=(1,2))
            )

        self.feature = nn.Sequential(
            block1,
            block2,
            block3,
            block4,
            block5,
            block6
            )

        # Reconv
        self.classification = nn.Sequential(
            nn.AvgPool2d((1,15),stride=1),
            nn.Conv2d(self.out_ch_tconv11,1,kernel_size=(self.sp_k_s,1),stride=(1,1), padding=(int(self.sp_k_s//2),0)),
            )

    def forward(self,x):
        x = self.feature(x)
        x = self.classification(x)

        return x

    def _dimension_cal(self, input_size, stride, k_size):
        return int((input_size-k_size)/stride+1)



