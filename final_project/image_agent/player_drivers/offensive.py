from image_agent.actuators.puck_actuator.models import load_model as load_puck_act
from image_agent.trajectory import Trajectory


class OffensiveHandler:
    def __init__(self, constants, karts_data, puck_data, i):
        self.puck_driver = load_puck_act()
        self.constants = constants
        self.kart_data = karts_data[i]
        self.puck_data = puck_data
        self.trajectory = None

    def reset(self):
        pass
        
    def act(self):
        pass
        
        