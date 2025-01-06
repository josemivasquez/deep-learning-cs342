from torchvision import transforms as T
from torchvision.transforms import functional as F

# class Compose(object):
#     def __init__(self, transforms):
#         self.transforms = transforms

#     def __call__(self, image, *args):
#         for t in self.transforms:
#             image, *args = t(image, *args)
#         return (image,) + tuple(args)
# class ToTensor(object):
#     def __call__(self, image, *args):
#         return (F.to_tensor(image),) + args


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, mask):
        for t in self.transforms:
            image, mask = t(image, mask)
        return image, mask
        
class ToTensor(object):
    def __call__(self, image, mask):
        return F.to_tensor(image), F.to_tensor(mask)