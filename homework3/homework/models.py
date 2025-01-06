import torch
import torch.nn.functional as F

import torch.nn as nn


class CNNClassifier(torch.nn.Module):
    def __init__(self, layers=(32, 64, 128)):
        super().__init__()
        """
        Your code here
        Hint: Base this on yours or HW2 master solution if you'd like.
        Hint: Overall model can be similar to HW2, but you likely need some architecture changes (e.g. ResNets)
        """
        i = 3
        o = layers[0]
        L = [
            torch.nn.Conv2d(i, o, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU()
        ]

        i = o
        for o in layers[1:]:
            L.append(self.Block(i, o, stride=2))
            i = o

        self.net = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(i, 6)

    class Block(torch.nn.Module):
        def __init__(self, channels_input, channels_output, stride=1):

            p = 0.2
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Dropout(p),
                torch.nn.Conv2d(channels_input, channels_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.BatchNorm2d(channels_output),
                torch.nn.ReLU(),

                torch.nn.Conv2d(channels_output, channels_output, kernel_size=1),
                torch.nn.BatchNorm2d(channels_output),
                # torch.nn.Dropout(p),
                torch.nn.ReLU(),

                torch.nn.Conv2d(channels_output, channels_output, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(channels_output),
                # torch.nn.Dropout(p),
                torch.nn.ReLU()
            )

            self.down_sample = None
            if channels_input != channels_output or stride != 1:
                self.down_sample = torch.nn.Sequential(
                    torch.nn.Conv2d(channels_input, channels_output, 1, stride=stride),
                    torch.nn.BatchNorm2d(channels_output)
                )

        def forward(self, x):

            identity = x
            if self.down_sample is not None:
                identity = self.down_sample(identity)

            return self.net(x) + identity

    def forward(self, x):
        mean = torch.mean(x, dim=(0, 2, 3)).to(x.device)
        std = torch.std(x, dim=(0, 2, 3)).to(x.device)

        x = (x - mean[None, :, None, None]) / std[None, :, None, None]
        
        z = self.net(x)
        z = z.mean((2, 3))
        return self.classifier(z)


class ExactConv(nn.Module):
    def __init__(self, i, o, up=False, ks=5, std=2, bn=True, nested=True):
        super().__init__()

        L = []
        conv = nn.Conv2d if not up else nn.ConvTranspose2d
        args = (i, o, ks, std, ks//2)
        kw = {} if not up else {'output_padding': 1}
        
        self.downsample = nn.Sequential(
            conv(*args, **kw), 
            nn.BatchNorm2d(o)
        )

        L.append(nn.Dropout(p=0.05))
        L.append(conv(*args, **kw))
          
        if bn:
            L.append(nn.BatchNorm2d(o))
        L.append(nn.ReLU())

        if not nested:
          self.net = nn.Sequential(*L)
          return
        
        # Nested Conv
        L.append(nn.Conv2d(o, o, 1))
        if bn:
            L.append(nn.BatchNorm2d(o))
        L.append(nn.ReLU())

        L.append(nn.Conv2d(o, o, 3, 1, 1))
        if bn:
            L.append(nn.BatchNorm2d(o))
        L.append(nn.ReLU())
        
        self.net = nn.Sequential(*L)

    def forward(self, x):
        identity = self.downsample(x)
        return self.net(x) + identity


class FCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        """
        Your code here.
        Hint: The FCN can be a bit smaller the the CNNClassifier since you need to run it at a higher resolution
        Hint: Use up-convolutions
        Hint: Use skip connections
        Hint: Use residual connections
        Hint: Always pad by kernel_size / 2, use an odd kernel_size
        """

        self.d1 = ExactConv(3, 8)
        self.d2 = ExactConv(8, 16)
        self.d3 = ExactConv(16, 32)
        self.d4 = ExactConv(32, 64)

        self.u1 = ExactConv(64, 32, up=True)
        self.u2 = ExactConv(32+32, 16, up=True)
        self.u3 = ExactConv(16+16, 8, up=True)
        self.u4 = ExactConv(8+8, 5, up=True)

        # self.downsample = nn.Conv2d(8, 32, 4, 4)
        # self.upsample = nn.Sequential(
        #   nn.ConvTranspose2d(64, 5, 16, 16, 0),
        #   nn.BatchNorm2d(5),
        #   nn.ReLU()
        # )


    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,H,W))
        @return: torch.Tensor((B,5,H,W))
        Hint: Apply input normalization inside the network, to make sure it is applied in the grader
        Hint: Input and output resolutions need to match, use output_padding in up-convolutions, crop the output
              if required (use z = z[:, :, :H, :W], where H and W are the height and width of a corresponding strided
              convolution
        """

        cat = lambda t1, t2: torch.cat((t1, t2), dim=1)
        
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        # res = self.upsample(d4)
        
        out = self.u1(d4)
        out = self.u2(cat(out, d3))
        out = self.u3(cat(out, d2))
        out = self.u4(cat(out, d1))
        
        # out = out + res
        return out


model_factory = {
    'cnn': CNNClassifier,
    'fcn': FCN,
}


def save_model(model, name=''):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            name = '%s' % n + name + '.th'
            path_file = path.join(path.dirname(path.abspath(__file__)), name)
            return save(model.state_dict(), path_file)

    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()

    name = '%s.th' % model
    path_file = path.join(path.dirname(path.abspath(__file__)), name)


    r.load_state_dict(load(path_file, map_location='cpu'))
    return r
