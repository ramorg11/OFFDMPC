import torch
import torch.nn as nn


class VelocityNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(VelocityNet, self).__init__()
        self.c0 = nn.Conv2d(in_ch, 64, 3, 1, 1)
        self.c1 = nn.Conv2d(64, 128, 4, 2, 1)
        self.c2 = nn.Conv2d(128, 256, 4, 2, 1)
        self.c3 = nn.Conv2d(256, 512, 4, 2, 1)
        self.c4 = nn.Conv2d(512, 512, 4, 2, 1)
        self.c5 = nn.Conv2d(512, 512, 4, 2, 1)
        self.c6 = nn.Conv2d(512, 512, 4, 2, 1)
        self.c7 = nn.Conv2d(512, out_ch, 4, 2, 1)

    def forward(self, x):
        hs = nn.functional.leaky_relu(self.c0(x))

        hs = nn.functional.leaky_relu(self.c1(hs))

        hs = nn.functional.leaky_relu(self.c2(hs))

        hs = nn.functional.leaky_relu(self.c3(hs))

        hs = nn.functional.leaky_relu(self.c4(hs))

        hs = nn.functional.leaky_relu(self.c5(hs))

        hs = nn.functional.leaky_relu(self.c6(hs))

        hs = nn.functional.leaky_relu(self.c7(hs))

        return hs