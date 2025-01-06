import torch
import torch.nn.functional as F
import torch.nn as nn


def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class ExactConv(nn.Module):
    def __init__(self, i, o, up=False, ks=3, std=2, nested=True, res=True):
        super().__init__()
        L = []
        conv = nn.Conv2d if not up else nn.ConvTranspose2d
        args = (i, o, ks, std, ks // 2)
        kw = {} if not up else {'output_padding': 1}

        self.down_sample = None if res else conv(*args, **kw)

        # L.append(nn.Dropout(p=0.05))
        L.append(conv(*args, **kw))
        L.append(nn.BatchNorm2d(o))
        L.append(nn.ReLU())

        if not nested:
            self.net = nn.Sequential(*L)
            return

        # Nested Conv
        L.append(nn.Conv2d(o, o, 3, 1, 1))
        L.append(nn.BatchNorm2d(o))
        L.append(nn.ReLU())

        L.append(nn.Conv2d(o, o, 3, 1, 1))
        L.append(nn.BatchNorm2d(o))
        L.append(nn.ReLU())

        self.net = nn.Sequential(*L)

    def forward(self, x):
        out = self.net(x)
        if self.down_sample is not None:
            identity = self.down_sample(x)
            out = out + identity

        return out

class Planner(torch.nn.Module):
    def __init__(self):
        super().__init__()

        """
        Your code here
        """
        layers = [16, 32, 64, 128]

        self.d1 = ExactConv(3, layers[0])
        self.d2 = ExactConv(layers[0], layers[1])
        self.d3 = ExactConv(layers[1], layers[2])
        self.d4 = ExactConv(layers[2], layers[3])

        self.u1 = ExactConv(layers[3], layers[2], up=True, nested=False, res=False)
        self.u2 = ExactConv(layers[2] * 2, layers[1], up=True, nested=False, res=False)
        self.u3 = ExactConv(layers[1] * 2, layers[0], up=True, nested=False, res=False)
        self.u4 = ExactConv(layers[0] * 2, 1, up=True, nested=False, res=False)
        

    def forward(self, img):
        """
        Your code here
        Predict the aim point in image coordinate, given the supertuxkart image
        @img: (B,3,96,128)
        return (B,2)
        """

        cat = lambda t1, t2: torch.cat((t1, t2), dim=1)

        d1 = self.d1(img)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)

        # res = self.upsample(d4)

        out = self.u1(d4)
        out = self.u2(cat(out, d3))
        out = self.u3(cat(out, d2))
        out = self.u4(cat(out, d1))

        # out = out + res

        out = torch.squeeze(out, dim=1)
        out = spatial_argmax(out)
        
        return out


def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Planner):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model():
    from torch import load
    from os import path
    r = Planner()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    from .controller import control
    from .utils import PyTux
    from argparse import ArgumentParser


    def test_planner(args):
        # Load model
        planner = load_model().eval()
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, planner=planner, max_frames=args.msteps, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser("Test the planner")
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-msteps', default=1000, type=int)
    args = parser.parse_args()
    test_planner(args)
