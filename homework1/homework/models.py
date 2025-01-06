import torch
import torch.nn.functional as F


class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here

        Compute mean(-log(softmax(input)_label))

        @input:  torch.Tensor((B,C))
        @target: torch.Tensor((B,), dtype=torch.int64)

        @return:  torch.Tensor((,))

        Hint: Don't be too fancy, this is a one-liner
        """
        batch_dist = F.softmax(input, dim=1)
        prob_batch = torch.take_along_dim(batch_dist, target.unsqueeze(1), dim=1).squeeze(1)
        return (-torch.log(prob_batch)).mean()


class LinearClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 3*64*64
        self.network = torch.nn.Linear(input_size, 6)

        # self.weights = torch.nn.Parameter(torch.zeros((6, 3, 64, 64), dtype=torch.float64))
        # self.bias = torch.nn.Parameter(torch.zeros((6,), dtype=torch.float64))

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.network(x.view(x.size(0), -1))

        # xw = (self.weights.unsqueeze(0) * x.unsqueeze(1)).sum((2, 3, 4))
        # return xw + self.bias.unsqueeze(0)


class MLPClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        input_size = 3*64*64
        hidden_size = 30

        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(hidden_size, 6)

        self.network = torch.nn.Sequential(self.linear1, self.activation, self.linear2)

    def forward(self, x):
        """
        Your code here

        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        return self.network(x.view(x.size(0), -1))


model_factory = {
    'linear': LinearClassifier,
    'mlp': MLPClassifier,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r
