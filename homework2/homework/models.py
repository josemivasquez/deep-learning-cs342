import torch


class CNNClassifier(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels_input, channels_output, stride=1):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Conv2d(channels_input, channels_output, kernel_size=3, padding=1, stride=stride),
                torch.nn.ReLU(),
                torch.nn.Conv2d(channels_output, channels_output, kernel_size=3, padding=1),
                torch.nn.ReLU()
            )

        def forward(self, x):
            return self.net(x)

    def __init__(self, layers=(8, 8, 16, 32)):
        """
        Your code here
        """
        i = 3
        o = layers[0]
        super().__init__()
        L = [
            torch.nn.Conv2d(i, o, kernel_size=7, padding=3, stride=2),
            torch.nn.ReLU(),
        ]

        i = o
        for o in layers[1:]:
            L.append(self.Block(i, o, stride=2))
            i = o

        self.net = torch.nn.Sequential(*L)
        self.classifier = torch.nn.Linear(i, 6)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        z = self.net(x)
        z = z.mean((2, 3))
        return self.classifier(z)


def save_model(model, name='cnn.th'):
    from torch import save
    from os import path
    if isinstance(model, CNNClassifier):
        return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), name))
    raise ValueError("model type '%s' not supported!"%str(type(model)))


def load_model(name='cnn.th'):
    from torch import load
    from os import path
    r = CNNClassifier()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), name), map_location='cpu'))
    return r
