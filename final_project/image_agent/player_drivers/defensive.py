from image_agent.actuators.kart_actuator.models import load_model as load_kart_act
from image_agent.trajectory import Trajectory

class DefensiveDriver:
    def __init__(self, constants, karts_data, puck_data, i):
        self.kart_driver = load_kart_act()
        self.trajectory = None
        self.constants = constants
    
    def reset(self, location):
        pass

    def act(self):
        pass