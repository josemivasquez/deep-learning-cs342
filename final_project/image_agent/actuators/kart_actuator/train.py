# from image_agent.runner import Runner
# from image_agent.trajectory import Trajectory
from .models import KartActuator
from .tester import Tester
from copy import deepcopy
import numpy as np
import torch
from image_agent.utils_coor import get2d

self.episode_length = 1000
class Hp:
    def __init__(self):
        self.lr = 0.2
        self.epochs= 1000
        self.sample = 16
        self.best_sample = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.03

class Trainer:
    def __init__(self):
        self.current = None
        self.n = None
        self.tester = Tester()

    def train(self, start, hp):
        actuator = start
        for e in range(hp.epochs):
            delta_sample = actuator.delta_sample()
            returns_minus = self.test(actuator, delta_sample, -hp.noise)
            returns_plus = self.test(actuator, delta_sample, hp.noise)
            sigma = np.array([returns_plus + returns_minus]).std()
            # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
            scores = {k: max(r_pos, r_neg) for k, (r_pos, r_neg) in enumerate(zip(returns_plus, returns_minus))}
            order = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)[:hp.best_sample]
            rollouts = [(returns_plus[k], returns_minus[k], delta_sample[k]) for k in order]
            actuator.update(rollouts, hp.lr, sigma, hp.best_sample)

            actuator.

    def test(self, actuator, delta_sample, noise):
        returns = []
        for delta in delta_sample:
            actuator.variating(delta * noise)
            response = self.tester.test(actuator)
            returns.append(response)
        
        return returns
        
def save_model(model):
    print('saving')
    torch.save(model, 'kart_actuator.pt')

if __name__ == '__main__':
    start = KartActuator(3)
    trainer = Trainer()
    trainer.train()
    save_model(trainer.current)