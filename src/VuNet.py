import torch
import torch.nn as nn


# Define the DecEnc model in PyTorch
class VuNet(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(VuNet, self).__init__()
        w = torch.nn.init.normal_
        self.c0 = nn.Conv2d(in_ch, 64, kernel_size=3, stride=1, padding=1)
        self.c1 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.c2 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.c3 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.c4 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.c5 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.c6 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.c7 = nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.c8 = nn.ConvTranspose2d(514, 512, kernel_size=4, stride=2, padding=1)
        self.c9 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.c10 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.c11 = nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1)
        self.c12 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.c13 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.c14 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.c15 = nn.Conv2d(64, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x, vres, wres):
        hs = nn.functional.leaky_relu(self.c0(x))
        hs = nn.functional.leaky_relu(self.c1(hs))
        hs = nn.functional.leaky_relu(self.c2(hs))
        hs = nn.functional.leaky_relu(self.c3(hs))
        hs = nn.functional.leaky_relu(self.c4(hs))
        hs = nn.functional.leaky_relu(self.c5(hs))
        hs = nn.functional.leaky_relu(self.c6(hs))
        hs = nn.functional.leaky_relu(self.c7(hs))
        z = torch.cat((hs, vres, wres), dim=1)
        h = nn.functional.relu(self.c8(z))
        h = nn.functional.relu(self.c9(h))
        h = nn.functional.relu(self.c10(h))
        h = nn.functional.relu(self.c11(h))
        h = nn.functional.relu(self.c12(h))
        h = nn.functional.relu(self.c13(h))
        h = nn.functional.relu(self.c14(h))
        h = torch.tanh(self.c15(h))
        return h
