import torch.nn as nn
import torch
from topkpool import TopkPool
from config import HyperParams


class CIA(nn.Module):
    def __init__(self):
        super(CIA, self).__init__()
        k = HyperParams['part'] + 1
        self.conv_sm = nn.Sequential(
            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, k, 1),
            nn.BatchNorm2d(k),
            nn.ReLU(True),
            nn.Softmax(dim=1),
        )
        self.pool = TopkPool()

    def forward(self, x):
        xpart = []
        M = self.conv_sm(x)
        _, c, _, _ = M.shape
        xf = torch.unsqueeze(M[:, 0, :, :], dim=1) * x
        xf = self.pool(xf)
        xpart.append(xf)
        for i in range(1, c):
            xt = torch.unsqueeze(M[:, i, :, :], dim=1) * x
            xtc = self.pool(xt)
            xpart.append(xtc)

        return xpart

