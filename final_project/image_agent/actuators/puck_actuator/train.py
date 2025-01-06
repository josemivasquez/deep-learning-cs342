from image_agent.runner import Runner
from image_agent.trajectory import Trajectory
from .models import PuckActuator

def get2d(vector):
    return np.array([v[0], v[2]])

def variate(model):
    for layer in model.layers:
        layer += torch.randn(*best_layer.shape)

class Trainer:
    def __init__(self, start):
        self.current = start

    def train(self, start, epochs):
        models = []
        current = start
        for e in range(epochs):
            models = self.generate(current)
            returns = self.act(models, test)
            best_indx = np.argmax(returns)
            current = models[best_indx]
        
        self.current = current    

    def act(self, models, tester):
        returns = []
        for model in models:
            response = tester.act(model)
            returns.append(response)
        
        return returns
        
    def generate(self, current):
        models = []
        for i in self.n:
            models.append(variate(deepcopy(model)))
        
        return models


if __name__ == '__main__':
    start = PuckActuator(3)
    trainer = Trainer(start, 30)