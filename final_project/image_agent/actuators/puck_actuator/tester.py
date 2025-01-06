class EmptyTeam:
    agent_type = 'state'
    def __init__(self):
        self.team = 1
        self.num_players = 0
    def act(self, player_state, opponent_state, soccer_state):
        [dict()] * self.num_players

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
        player_position = get_2d(player_state[0]['location'])
        player_velocity = get_2d(player_position[0]['velocity'])
        puck_position = get_2d(soccer_state[0]['location'])

        if self.aim_point is not None:
            reward = - (self.aim_point - player_position) ** 2
        self.response += reward

        self.aim_point = self.trajectory(player_position)
        action = self.model.act(puck_position - player_position, player_velocity, aim_point)

        return action

    def get_return(self):
        return self.response

class Tester:
    def __init__(self):
        # Initial Conditions
        self.trajectory = Trajectory((0, 0), (20, 20))
        self.empty_team = EmptyTeam()
        self.test_team = TestTeam()

        observers = (0,)
        nframes = 500
        self.runner = Runner(test_team, empty_team, (10, 10), nframes, observers, False, False)
    
    def reset(self):
        runner.ws.set_ball_location((0, 0, 0))
        runner.ws.set_kart_location(0, (0, 0, -20))
    
    def test(self, model):
        self.reset()
        test_team.reset(model, self.trajectory)
        runner.play()
        return test_team.get_return()