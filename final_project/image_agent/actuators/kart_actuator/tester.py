from image_agent.trajectory import Trajectory
from image_agent.runner import Runner
from image_agent.utils_coor import get2d
import numpy as np

class Tester:
    def __init__(self):
        # Initial Conditions
        self.trajectory = Trajectory((0, 0), (40, 40))
        self.empty_team = EmptyTeam()
        self.test_team = TestTeam()

        observers = (0,)
        nframes = 500
        self.runner = Runner(
          self.test_team, self.empty_team, (10, 10), 
          nframes, observers, False, False
        )
    
    def reset(self):
        self.runner.race.start()
        self.runner.ws.update()
        self.runner.ws.set_ball_location((0, 0, 0))
        self.runner.ws.set_kart_location(0, (0, 0, -2))
    
    def test(self, model):
        self.reset()
        self.test_team.reset(model, self.trajectory)
        self.runner.play()
        return self.test_team.get_return()

class EmptyTeam:
    agent_type = 'state'
    def __init__(self):
        self.team = 1
        self.num_players = 0
    def act(self, player_state, opponent_state, soccer_state):
        return [dict()] * self.num_players

class TestTeam:
    agent_type = 'state'
    def __init__(self):
        self.team = 0
        self.num_players = 1
        
        self.response = None
        self.trajectory = None
        self.model = None
        self.aim_point = None
    
    def reset(self, model, trajectory):
        self.model = model
        self.response = 0
        self.trajectory = trajectory
    
    def act(self, player_state, opponent_state, soccer_state):
        kart_state = player_state[0]['kart']
        player_position = get2d(kart_state['location'])
        player_velocity = get2d(kart_state['velocity'])
        puck_position = get2d(soccer_state['ball']['location'])

        if self.aim_point is not None:
            reward = - np.linalg.norm(self.aim_point - puck_position)
            self.response += reward

        self.aim_point = self.trajectory.next(player_position)
        action = self.model.act(puck_position - player_position, player_velocity, self.aim_point)

        return [action]

    def get_return(self):
        return self.response

