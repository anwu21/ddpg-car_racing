import torch
import torch.nn as nn

class dcgan_conv(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_conv, self).__init__()
        self.main = nn.Sequential(
                nn.Conv2d(nin, nout, 4, 2, 1),
                nn.BatchNorm2d(nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)

class dcgan_fc(nn.Module):
    def __init__(self, nin, nout):
        super(dcgan_fc, self).__init__()
        self.main = nn.Sequential(
                nn.Linear(nin, nout),
                nn.LeakyReLU(0.2, inplace=True),
                )

    def forward(self, input):
        return self.main(input)
    
class Actor(nn.Module):
    def __init__(self, dim=64, nc=3):
        super(Actor, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 96 x 96
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 48 x 48
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 24 x 24
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 12 x 12
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 6 x 6
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 3 x 3
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 2, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.fc1 = dcgan_fc(dim * 2 * 2, 256)
        self.fc2 = dcgan_fc(256, 16)
        self.fc3 = nn.Sequential(
                nn.Linear(16, 3),
                #nn.BatchNorm1d(dim),
                nn.Tanh()
                )

    def forward(self, state):
        h = self.c1(state)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = self.c5(h)
        h = self.c6(h)
        h = self.fc1(h.view(-1, self.dim * 2 * 2))
        h = self.fc2(h)
        return self.fc3(h)

class Critic(nn.Module):
    def __init__(self, dim=64, nc=3):
        super(Critic, self).__init__()
        self.dim = dim
        nf = 64
        # input is (nc) x 96 x 96
        self.c1 = dcgan_conv(nc, nf)
        # state size. (nf) x 48 x 48
        self.c2 = dcgan_conv(nf, nf * 2)
        # state size. (nf*2) x 24 x 24
        self.c3 = dcgan_conv(nf * 2, nf * 4)
        # state size. (nf*4) x 12 x 12
        self.c4 = dcgan_conv(nf * 4, nf * 8)
        # state size. (nf*8) x 6 x 6
        self.c5 = dcgan_conv(nf * 8, nf * 8)
        # state size. (nf*8) x 3 x 3
        self.c6 = nn.Sequential(
                nn.Conv2d(nf * 8, dim, 2, 1, 0),
                nn.BatchNorm2d(dim),
                nn.Tanh()
                )
        self.action_transform = nn.Sequential(nn.Linear(3, 2*2*16),
                                             nn.Tanh(),
                                             nn.Linear(2*2*16, 2*2*16),
                                             nn.Tanh()) 
        
        self.fc1 = dcgan_fc(dim * 2 * 2 + 64, 256)
        self.fc2 = dcgan_fc(256, 16)
        self.fc3 = nn.Sequential(
                nn.Linear(16, 1),
                nn.Tanh()
                )


    def forward(self, state, action):
        h = self.c1(state)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = self.c5(h)
        h = self.c6(h)
        h = h.view(-1, self.dim * 2 * 2)
        
        z_action = self.action_transform(action).view(-1, 64)
        h = torch.cat([h, z_action], 1)
        h = self.fc1(h)
        h = self.fc2(h)
        return self.fc3(h)
    
