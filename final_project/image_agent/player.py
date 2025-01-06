# from .models import load_model
from .puck_detector.models_ramiro import load_model as load_detector
from .player_drivers.defensive import DefensiveDriver
from .player_drivers.offensive import OffensiveHandler
import numpy as np

TEAM0GOAL = np.array([0, 70])
TEAM1GOAL = np.array([0, -70])

def to2d(arr):
    return np.array(
        [arr[0], arr[2]]
    )

def feature_extractor(player_states):
    locations = []
    velocities = []
    fronts = []

    for state in player_states:
        velocities.append(
            to2d(state['velocity'])
        )
        locations.append(
            to2d(state['location'])
        )
        fronts.append(
            to2d(state['front'])
        )
    
    return locations, velocities, fronts

def do_constants(constants, team):
    if team == 0:
        constants['local_goal'] = TEAM0GOAL
        constants['enemy_goal'] = TEAM1GOAL
    else:
        constants['local_goal'] = TEAM1GOAL
        constants['enemy_goal'] = TEAM0GOAL


class Team:
    def __init__(self):
        self.drivers_dict = {}
        self.current_drivers = []

        self.puck_detector = load_detector()
        
        self.team = None
        self.num_players = None
        
        self.constants = {}
        self.karts_data = []
        self.puck_data = []
        
    
    def update_karts_data(self):
        locations, velocities, fronts = feature_extractor(player_states)
        i = 0
        for loc, vel, front in zip(locations, velocities, fronts):
            kart_data = self.data[i]
            kart_data['loc'] = loc
            kart_data['vel'] = vel
            kart_data['front'] = front
            i += 1
    
    def new_match(self, team, num_players):
        self.team = team
        self.num_players = num_players
        self.do_constants()
        for i in num_players:
            kart_dict = dict()
            kart_dict[f'{i}_off'] = OffensiveDriver(self.constants, self.karts_data, self.puck_data, i)
            kart_dict[f'{i}_def'] = DefensiveDriver(self.constants, self.karts_data, self.puck_data, i)
            self.drivers_dict.append(kart_dict)

    def holding(self, i):
        puck_location = self.puck_data[0]
        kart_location = self.karts_data[i]['loc']
        
        if np.norm(puck_location - kart_location) < 5:
            return True
        else:
            return False

    def visible(self, prob):
        return self.puck_data[1] > 0.3
    
    def near(self):
        pass

    def transition(self, holding, visible):
        if holding:
            return 'offensive'
        else:
            return 'defensive'

    def update_puck_estim(self, locations, visions):
        estimations = []
        probabilities = []
        for vision, location in zip(visions, locations):
            estimation, probability = location + self.puck_detector.detect(vision)
            estimations.append(probability)
            probabilities.append(probability)
        
        estimations = np.array(estimations)
        probabilities = np.array(probabilities)
        prob = np.mean(probabilities)

        response = (estimations * probabilities) / sum(probabilities)
        self.puck_data[0] = response
        self.puck_data[1] = prob

    def act(self, player_state, player_image):
        # Visibility for team, holding per kart
        self.update_karts_data(player_state)
        self.update_puck_estim(locations, player_image)

        visible = self.visible()
        actions = []
        for i, current_driver, kart_driver_dict in enumerate(zip(self.current_drivers, self.drivers_dict)):
            holding = self.holding(i)
            new_key = self.transition(holding, visible)
            if kart_driver_dict[new_key] == current_driver:
                actions.append(current_driver.act())
                continue

            self.current_drivers[i] = kart_driver_dict[new_key]
            self.current_drivers[i].reset()
            actions.append(self.current_drivers[i].act())
        
        return actions

# import numpy as np

# from .utils_image import indices2imagecoor, abscoor2imagecoor, imagecoor2indices
# from torchvision.transforms import functional as F


# GOAL_0_ABS_POSITION = [0, 0, 70]
# GOAL_1_ABS_POSITION = [0, 0, -70]
# norm = np.linalg.norm

# from time import time

# def control(aim_point, current_vel, kart_position):
#     action = {}
#     acceleration = 0
#     brake = False
#     steer = 0
#     drift = False
#     nitro = False

#     screen_size = np.array([400, 300])
#     kart_position = np.array([0, 0.125])

#     # Steer
#     normalized_vector = (aim_point - kart_position) * screen_size
#     steer = - np.arctan(normalized_vector[0] / normalized_vector[1]) * (2 / np.pi)
    
#     # target_velocity = 40
#     target_velocity = 55/abs(steer)**0.08
    
#     # Drift and Nitro
#     drift_limit = 0.5
#     if drift_limit < abs(steer):
#         drift = True
    
#     nitro_limit = 0.2
#     if abs(steer) < nitro_limit:
#         nitro = True
    
#     # Brake
#     brake_limit = 10
#     if current_vel - target_velocity > brake_limit or abs(steer)>0.4:
#         brake = True
    
#     # Acceleration(cv, tv, steer)
#     # acceleration_limit = 20
#     if current_vel < target_velocity:# and abs(steer)<0.45:
#         acceleration = (target_velocity - current_vel) / target_velocity #*(1-abs(steer))**2
#     elif current_vel < 5:
#         acceleration = (target_velocity - current_vel) / target_velocity 
 
#     steer *= 1.5
    
#     action = {
#     'acceleration' : acceleration,
#     'steer' : steer,
#     'drift' : drift,
#     'brake' : brake,
#     'nitro' : nitro,
#     }

#     return action

# def heatmap2indmaxpair(puck_heatmap):
#     ind_max = np.argmax(puck_heatmap)
#     return np.array([ind_max // puck_heatmap.shape[1], ind_max % puck_heatmap.shape[1]])

# def inverse_sigmoid(x):
#     return np.log(x) - np.log(1 - x)

# class Team:
#     agent_type = 'image'

#     def __init__(self):
#         """
#           TODO: Load your agent here. Load network parameters, and other parts of our model
#           We will call this function with default arguments only
#         """
#         self.team = None
#         self.num_players = None

#         self.puck_viewer = load_model_ramiro()
#         self.puck_viewer.eval()

#         self.center = np.array([0, 0, 0])

#         # self.pred_logit_threshold = inverse_sigmoid(0.5)
#         self.pred_logit_threshold = -4
        
#         self.current_puck_pred = None
#         self.goal_coor = None
#         self.shape = None

#         self.crash_steps_back = 50
        
#         self.on_crash_counter = None
#         self.reverse = None
#         self.on_goal = None
#         self.crash_type = None
#         self.aim_reverse = None

#         self.space_limits = np.array([45, 10, 63])
#         self.limit_margin = np.array([4, 4, 4])
#         self.r_puck = 0.05

#         self.to_log = {}
#         self.to_circle_draw = []

#         self.frame = 0
#         self.predict_each = 3
#         self.last_heatmap = None

#     def new_match(self, team: int, num_players: int) -> list:
#         """
#         Let's start a new match. You're playing on a `team` with `num_players` and have the option of choosing your kart
#         type (name) for each player.
#         :param team: What team are you playing on RED=0 or BLUE=1
#         :param num_players: How many players are there on your team
#         :return: A list of kart names. Choose from 'adiumy', 'amanda', 'beastie', 'emule', 'gavroche', 'gnu', 'hexley',
#                  'kiki', 'konqi', 'nolok', 'pidgin', 'puffy', 'sara_the_racer', 'sara_the_wizard', 'suzanne', 'tux',
#                  'wilber', 'xue'. Default: 'tux'
#         """
#         """
#            TODO: feel free to edit or delete any of the code below
#         """
#         self.team, self.num_players = team, num_players

#         # Default value for run
#         self.current_puck_pred = [(0, 0)] * self.num_players
#         self.goal_coor = [(0, 0)] * self.num_players
        
#         # Crash Atts
#         self.on_crash_counter = [0] * self.num_players
#         self.reverse = [False] * self.num_players
#         self.on_goal = [None] * self.num_players
#         self.crash_type = [None] * self.num_players
#         self.aim_reverse = [None] * self.num_players

#         return ['tux'] * num_players
    
#     def get_aim_point(self, goal_coor, puck_coor, i):
#         if i == 0:
#             self.to_log['goal'] = goal_coor
#             self.to_log['puck'] = puck_coor
        
#         vector = goal_coor - puck_coor
#         vector /= self.shape
#         vector[1] = -vector[1]

#         alpha = np.arctan2(vector[1], vector[0])

#         if i == 0:
#             self.to_log['vector'] = str(vector)
#             self.to_log['alpha'] = str(alpha * (180 / np.pi))

#         move = np.cos(alpha) * self.r_puck
        
#         return puck_coor - np.array([move, 0])
    
#     def get_puck_heatmap(self, vision):
#         vision = torch.Tensor(vision)
#         vision = vision[None, :]
#         vision = vision.movedim(-1, 1)

#         puck_heatmap = self.puck_viewer(vision)
#         puck_heatmap = puck_heatmap[0][0].detach().numpy()

#         return puck_heatmap
    
#     def crash_query(self, velocity, location):
#         return np.isclose(velocity, 0, rtol=0, atol=10).__and__(
#             any(abs(location) > self.space_limits)
#         )
    
#     def finish_back_query(self, location, i):
#         return (self.on_crash_counter[i] == 0).__or__(
#             all(abs(location) < (self.space_limits - self.limit_margin))
#         )
    
#     def forward_action(self, vision, state, velocity_norm, i) -> dict:
#         if self.frame % self.predict_each == 0:
#             puck_heatmap = self.get_puck_heatmap(vision)
#             self.last_heatmap = puck_heatmap

#         else:
#             puck_heatmap = self.last_heatmap

#         # pred_time = time()
#         indmax = heatmap2indmaxpair(puck_heatmap)
#         logit = puck_heatmap[indmax[0]][indmax[1]]
#         self.to_log['logit'] = logit
        
#         if self.pred_logit_threshold < logit:
#             puck_coor = indices2imagecoor(indmax, self.shape)
#             # Only when see it update the current
#             self.current_puck_pred[i] = indmax

#         else:
#             # Don't see puck
#             puck_coor = indices2imagecoor(self.current_puck_pred[i], self.shape)

#         # Getting goal image_coor
#         goal_coor = abscoor2imagecoor(
#             np.array(state['camera']['projection']),
#             np.array(state['camera']['view']),
#             np.array(GOAL_0_ABS_POSITION if self.team == 0 else GOAL_1_ABS_POSITION)
#         )

#         self.goal_coor[i] = imagecoor2indices(goal_coor, self.shape)
#         aim_point = self.get_aim_point(goal_coor, puck_coor, i)
#         if i == 0:
#             self.to_log['aim_point'] = str(aim_point)
#         if i == 0:
#             self.to_circle_draw.append((imagecoor2indices(aim_point, self.shape), 'blue'))

#         action = control(aim_point, velocity_norm, np.array([0, 0]))
#         return action
    
#     def back_controller(self, crash_type, front, location, aim_point) -> dict:
#         action = {}
#         if crash_type:
#             action['brake'] = True
#             action['acceleration'] = 0
#             v = -front
        
#         else:
#             action['brake'] = False
#             action['acceleration'] = 0.5
#             v = front
        
#         m = np.cross(v, (aim_point - location))
#         steer = np.arcsin(
#             (norm(m) * np.sign(m[2])) / (norm(aim_point - location))
#         ) * (2 / np.pi)
#         steer *= 1.2

#         action['steer'] = steer

#         return action
        
#     def back_action(self, front, location, i) -> dict:
#         self.on_crash_counter[i] = self.crash_steps_back
#         self.reverse[i] = True
#         on_goal = self.get_reverse_on_goal(location)
#         crash_type = self.get_crash_type(location, front)
#         aim_point = self.get_reverse_aim_point(location, on_goal, front)
        
#         return self.back_controller(crash_type, front, location, aim_point)

#     def get_reverse_on_goal(self, location):
#         return np.abs(location)[2] > np.abs(GOAL_0_ABS_POSITION[2])
    
#     def get_reverse_aim_point(self, location, on_goal, front):
#         if on_goal:
#             if norm(location - GOAL_0_ABS_POSITION) < norm(location - GOAL_1_ABS_POSITION):
#                 aim_point = GOAL_0_ABS_POSITION
#             else:
#                 aim_point = GOAL_1_ABS_POSITION
        
#         else:
#             aim_point = location - front
        
#         return aim_point

#     def get_crash_type(self, location, front):
#         return norm(location + front - self.center).__gt__(
#             norm(location - front - self.center)
#         )

#     def get_action(self, state, vision, i):
#         velocity_norm = norm(state['kart']['velocity'])
#         location = np.array(state['kart']['location'])
#         front = np.array(state['kart']['front'])

#         # Crashing Handle
#         if not self.reverse[i]:
#             self.to_log['direction'] = 'forward'
#             crashed = self.crash_query(velocity_norm, location)
#             if not crashed:
#                 return self.forward_action(vision, state, velocity_norm, i)
#             else:
#                 return self.back_action(front, location, i)

#         else:
#             self.to_log['direction'] = 'reverse'
#             finish_back = self.finish_back_query(location, i)
#             if not finish_back:
#                 return self.back_action(front, location, i)
                
#             else:
#                 self.reverse[i] = False
#                 self.on_crash_counter[i] = 0

#                 return self.forward_action(vision, state, velocity_norm, i)


#     def act(self, player_state, player_image):
#         """
#         This function is called once per timestep. You're given a list of player_states and images.

#         DO NOT CALL any pystk functions here. It will crash your program on your grader.

#         :param player_state: list[dict] describing the state of the players of this team. The state closely follows
#                              the pystk.Player object <https://pystk.readthedocs.io/en/latest/state.html#pystk.Player>.
#                              See HW5 for some inspiration on how to use the camera information.
#                              camera:  Camera info for each player
#                                - aspect:     Aspect ratio
#                                - fov:        Field of view of the camera
#                                - mode:       Most likely NORMAL (0)
#                                - projection: float 4x4 projection matrix
#                                - view:       float 4x4 view matrix
#                              kart:  Information about the kart itself
#                                - front:     float3 vector pointing to the front of the kart
#                                - location:  float3 location of the kart
#                                - rotation:  float4 (quaternion) describing the orientation of kart (use front instead)
#                                - size:      float3 dimensions of the kart
#                                - velocity:  float3 velocity of the kart in 3D

#         :param player_image: list[np.array] showing the rendered image from the viewpoint of each kart. Use
#                              player_state[i]['camera']['view'] and player_state[i]['camera']['projection'] to find out
#                              from where the image was taken.

#         :return: dict  The action to be taken as a dictionary. For example `dict(acceleration=1, steer=0.25)`.
#                  acceleration: float 0..1
#                  brake:        bool Brake will reverse if you do not accelerate (good for backing up)
#                  drift:        bool (optional. unless you want to turn faster)
#                  fire:         bool (optional. you can hit the puck with a projectile)
#                  nitro:        bool (optional)
#                  rescue:       bool (optional. no clue where you will end up though.)
#                  steer:        float -1..1 steering angle
#         """
#         # Capture the shape
        

#         self.to_circle_draw.clear()
#         self.to_log.clear()
        
#         actions = []
#         i = 0
#         for state, vision in zip(player_state, player_image):
#             # vision = F.resize(F.to_tensor(vision), (300//3, 400//3)) # 3 @ H @ W

#             if self.shape is None:
#                 self.shape = np.array(vision.shape[0:2])

#             total_time = time()
#             action = self.get_action(state, vision, i)
#             total_time = time() - total_time
#             # print(f'agent {i}, total time: {total_time}')
#             actions.append(action)
#             if i == 0 and self.team == 0:
#                 self.to_log[f'location, player{i}, team{self.team}'] = state['kart']['location']

#             i += 1

#             self.to_log['frame'] = self.frame
#             self.to_circle_draw.append((self.current_puck_pred[0], 'red'))
#             self.to_circle_draw.append((self.goal_coor[0], 'green'))

#         self.frame += 1
#         return actions

#     def get_draw_info(self):
#         return self.to_log, self.to_circle_draw
