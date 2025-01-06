import torch.nn as nn

class PuckActuator():
    def __init__(self, n_layers, input_features=8, output_features=2):
        
        self.n_layers = n_layers
        layers = [nn.Linear(input_features, 32)]
        for i in range(1, n_layers-1):
            layers.append(nn.Linear(32, 32))
        layers.append(nn.Linear(32, output_features))

        setattr(self, f'l_0', nn.Linear(input_features, 32))
        for i in range(1, layers - 1):
            setattr(self, f'l_{i}', nn.Linear(32, 32))
        
        setattr(self, f'l_{layers-1}', nn.Linear(32, output_features))

    def forward(self, x):
        for l in layers:
            x = l(x)
        return x
    
    def act(self, player_position, player_velocity, puck_position, aim_point):
        input_tensor = torch.Tensor(
            list(player_position) + list(player_velocity) + list(puck_position) + list(aim_point)
        )
        out = self.forward(input_tensor).detach().numpy()
        acc, steer = out
        
        return acc, steer
  
def load_model():
    pass


