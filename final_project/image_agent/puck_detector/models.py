import torch
import torch.nn.functional as F

def spatial_argmax(logit):
    """
    Compute the soft-argmax of a heatmap
    :param logit: A tensor of size BS x H x W
    :return: A tensor of size BS x 2 the soft-argmax in normalized coordinates (-1 .. 1)
    """
    weights = F.softmax(logit.view(logit.size(0), -1), dim=-1).view_as(logit)
    return torch.stack(((weights.sum(1) * torch.linspace(-1, 1, logit.size(2)).to(logit.device)[None]).sum(1),
                        (weights.sum(2) * torch.linspace(-1, 1, logit.size(1)).to(logit.device)[None]).sum(1)), 1)

class Detector(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.Conv2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                      stride=stride)
            self.c2 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.c3 = torch.nn.Conv2d(n_output, n_output, kernel_size=kernel_size, padding=kernel_size // 2)
            self.b1 = torch.nn.BatchNorm2d(n_output)
            self.b2 = torch.nn.BatchNorm2d(n_output)
            self.b3 = torch.nn.BatchNorm2d(n_output)
            self.skip = torch.nn.Conv2d(n_input, n_output, kernel_size=1, stride=stride)

        def forward(self, x):
            return F.relu(self.b3(self.c3(F.relu(self.b2(self.c2(F.relu(self.b1(self.c1(x)))))))) + self.skip(x))

    class UpBlock(torch.nn.Module):
        def __init__(self, n_input, n_output, kernel_size=3, stride=2):
            super().__init__()
            self.c1 = torch.nn.ConvTranspose2d(n_input, n_output, kernel_size=kernel_size, padding=kernel_size // 2,
                                               stride=stride, output_padding=1)

        def forward(self, x):
            return F.relu(self.c1(x))

    def __init__(self, layers=[16, 32], n_class=1, kernel_size=3, use_skip=True):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()

        c = 3
        self.use_skip = use_skip
        self.n_conv = len(layers)
        skip_layer_size = [3] + layers[:-1]
        for i, l in enumerate(layers):
            self.add_module('conv%d' % i, self.Block(c, l, kernel_size, 2))
            c = l
        # Produce lower res output
        for i, l in list(enumerate(layers))[::-1]:
            self.add_module('upconv%d' % i, self.UpBlock(c, l, kernel_size, 2))
            c = l
            if self.use_skip:
                c += skip_layer_size[i]
        self.classifier = torch.nn.Conv2d(c, n_class, 1)

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """

        mean = torch.mean(x, dim=(0, 2, 3)).to(x.device)
        std = torch.std(x, dim=(0, 2, 3)).to(x.device)
        z = (x - mean[None, :, None, None]) / std[None, :, None, None]

        up_activation = []
        for i in range(self.n_conv):
            up_activation.append(z)
            z = self._modules['conv%d' % i](z)

        for i in reversed(range(self.n_conv)):
            z = self._modules['upconv%d' % i](z)
            # Fix the padding
            z = z[:, :, :up_activation[i].size(2), :up_activation[i].size(3)]

            # Add the skip connection
            if self.use_skip:
                z = torch.cat([z, up_activation[i]], dim=1)

        # print(z.shape)
        z = self.classifier(z)
        # print(z.shape)

        # z = z.mean(dim=1)
        # z = spatial_argmax(z)

        return z        

    # def detect(self, image, **kwargs):
    #     """
    #        Your code here.
    #        Implement object detection here.
    #        @image: 3 x H x W image
    #        @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
    #                 return no more than 30 detections per image per class. You only need to predict width and height
    #                 for extra credit. If you do not predict an object size, return w=0, h=0.
    #        Hint: Use extract_peak here
    #        Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
    #              scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
    #              out of memory.
    #     """
    #     cls, size = self.forward(image[None])
    #     size = size.cpu()
    #     return [[(s, x, y, float(size[0, 0, y, x]), float(size[0, 1, y, x]))
    #              for s, x, y in extract_peak(c, max_det=30, **kwargs)] for c in cls[0]]

def save_model(model):
    from torch import save
    from os import path
    if isinstance(model, Detector):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'planner.th'))
    raise ValueError("model type '%s' not supported!" % str(type(model)))

def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'planner.th'), map_location='cpu'))
    return r
