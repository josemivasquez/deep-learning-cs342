import torch
import torch.nn.functional as F
import torch.nn as nn


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    """
       Your code here.
       Extract local maxima (peaks) in a 2d heatmap.
       @heatmap: H x W heatmap containing peaks (similar to your training heatmap)
       @max_pool_ks: Only return points that are larger than a max_pool_ks x max_pool_ks window around the point
       @min_score: Only return peaks greater than min_score
       @return: List of peaks [(score, cx, cy), ...], where cx, cy are the position of a peak and score is the
                heatmap value at the peak. Return no more than max_det peaks per image
    """

    width = heatmap.size(1)
    flatten2coord = lambda flatten : (flatten // width, flatten % width)

    pool = F.max_pool2d(heatmap[None, None, :], max_pool_ks,
                        1, (max_pool_ks - 1) // 2)[0][0]
                        
    flatten_peaks = torch.logical_and(heatmap == pool, pool > min_score).view(-1)
    flatten_heatmap = heatmap.view(-1)

    flatten_heatmap[~flatten_peaks] = float('-inf')
    top_peaks = flatten_heatmap.topk(min(max_det, flatten_peaks.count_nonzero()))

    response = []
    for value, f_index in zip(*top_peaks):
        value = float(value)
        f_index = int(f_index)

        cy, cx = flatten2coord(f_index)
        response.append( (value, cx, cy) )

    return response


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

class Detector(torch.nn.Module):
    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        n_class = 3

        layers = [16, 32, 64, 128]

        self.d1 = ExactConv(3, layers[0])
        self.d2 = ExactConv(layers[0], layers[1])
        self.d3 = ExactConv(layers[1], layers[2])
        self.d4 = ExactConv(layers[2], layers[3])

        # ------------------- COPY -------------------
        self.u1 = ExactConv(layers[3], layers[2], up=True, nested=False, res=False)
        self.u2 = ExactConv(layers[2] * 2, layers[1], up=True, nested=False, res=False)
        self.u3 = ExactConv(layers[1] * 2, layers[0], up=True, nested=False, res=False)
        self.u4 = ExactConv(layers[0] * 2, n_class, up=True, nested=False, res=False)
        # ------------------- FINISH COPY ------------


    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
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

    def detect(self, image, **kwargs):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """

        with torch.no_grad():
            peaks = self(image[None, :])[0]
          
        response = []
        for peak in peaks:
            extracted = extract_peak(peak, max_det=30, **kwargs)
            extracted = [e + (float(0), float(0)) for e in extracted]
            response.append(extracted)

        return tuple(response)


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset

    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()
