import torch
import torch.nn as nn
import torch.nn.functional as F

class KartActuator():
    def __init__(self, n_layers, input_features=6, output_features=2):
        self.n_layers = n_layers
        layers = [nn.Linear(input_features, 32)]
        for i in range(1, n_layers-1):
            layers.append(nn.Linear(32, 32))
        layers.append(nn.Linear(32, output_features))

        self.layers = layers

        # setattr(self, f'l_0', nn.Linear(input_features, 32))
        # for i in range(1, n_layers - 1):
        #     setattr(self, f'l_{i}', nn.Linear(32, 32))
        
        # setattr(self, f'l_{layers-1}', nn.Linear(32, output_features))

    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x

    def act(self, puck_position, player_velocity, aim_point):
        input_tensor = torch.Tensor(
            list(puck_position) + list(player_velocity) + list(aim_point)
        )
        out = self.forward(input_tensor).detach().numpy()
        acc, steer = out

        action = {
          'acceleration' : acc,
          'steer' : steer
        }
        return action

class KartLinearActuator:
    def __init__(self, n_inputs, n_outputs):
        self.theta = torch.zeros((n_outputs, n_inputs), requires_grad=False)
        self.delta = None
    
    def delta_sample(self, n):
        response = []
        for i in range(n):
            response.append(torch.randn(*self.theta.shape))
        return response
        
    def update(self, rollouts, lr, sigma, n):
        with torch.no_grad():
            step = torch.zeros(*self.theta.shape)
            for r_pos, r_neg, d in rollouts:
                step += (r_pos - r_neg) * d
            self.theta += lr / (n * sigma) * step
    
    def variating(self, delta):
        self.delta = delta

    def act(self, puck_loc, player_vel, aim_point):
        puck_loc = torch.Tensor(puck_loc)
        player_vel = torch.Tensor(player_vel)
        aim_point = torch.Tensor(aim_point)

        F.normalize(puck_loc, dim=0, out=puck_loc)
        F.normalize(player_vel, dim=0, out=aim_point)
        F.normalize(aim_point, dim=0, out=aim_point)

        input_tensor = torch.cat(
            (puck_loc, player_vel, aim_point)
        )

        if self.delta is not None:
            out = F.Linear(input_tensor, self.theta + self.delta)
        else:
            out = F.Linear(input_tensor, self.theta)
        
        steer, acc = out
        return steer, acc

def load_model():
    pass




