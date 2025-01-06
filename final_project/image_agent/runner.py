import pystk
import numpy as np
from argparse import ArgumentParser
from typing import Optional, List, Tuple
from .runner_utils.data_saver import DataSaver
from .runner_utils.video_saver import VideoSaver
from .runner_utils.utils_game import *
from .runner_utils.teams_constant import TEAM_DICT
from .utils_coor import *

import traceback

TRACK = 'icy_soccer_field'
MODE = pystk.RaceConfig.RaceMode.SOCCER
PUCK_OBJECT_TYPE = 8

def do_graphics_config(resolution):
    graphics_config = pystk.GraphicsConfig.hd()
    graphics_config.screen_width, graphics_config.screen_height = resolution
    return graphics_config

def do_race_config(team1, team2):
    controller_object = pystk.PlayerConfig.Controller
    if not hasattr(type(team1), 'dummy'):
        team1_controller = controller_object.PLAYER_CONTROL
    else:
        team1_controller = controller_object.AI_CONTROL
    
    if not hasattr(type(team2), 'dummy'):
        team2_controller = controller_object.PLAYER_CONTROL
    else:
        team2_controller = controller_object.AI_CONTROL

    race_config = pystk.RaceConfig()
    race_config.num_kart = team1.num_players + team2.num_players

    race_config.players.clear()
    for i in range(team1.num_players):
        race_config.players.append(
            pystk.PlayerConfig(
                controller=team1_controller,
                team=0
            )
        )

    for i in range(team2.num_players):
        race_config.players.append(
            pystk.PlayerConfig(
                controller=team2_controller,
                team=1
            )
        )

    race_config.mode = MODE
    race_config.track = TRACK

    return race_config

class Runner(object):
    def __init__(
        self, team1, team2, resolution, n_frames, observers, 
        save_video, save_td, video_args=(), td_args=()
    ):
        self.team1 = team1
        self.team2 = team2
        
        graphics_config = do_graphics_config(resolution)
        pystk.init(graphics_config)

        race_config = do_race_config(team1, team2)
        self.race = pystk.Race(do_race_config(team1, team2))

        # State of the game
        self.ws = pystk.WorldState()
        self.ws.update()

        self.data_saver = DataSaver()
        self.video_saver = VideoSaver()

        # Control Atributes
        self.record_td = save_td
        self.record_video = save_video
        self.observers = observers
        self.n_frames = n_frames

        # For log
        self.print_each = 400

        # For draw
        self.draw_info_t1 = 'get_draw_info' in dir(self.team1)
        self.draw_info_t2 = 'get_draw_info' in dir(self.team2)

    def play(self):
        to_video = []
        to_td = []

        self.race.start()
        self.race.step()
        self.ws.update()

        max_time1 = 0
        max_time2 = 0
        for i in range(self.n_frames):
            team1_actions, time1 = self.get_actions(self.team1)
            team2_actions, time2 = self.get_actions(self.team2)
            all_actions = team1_actions + team2_actions
            self.race.step(all_actions)
            self.ws.update()

            for obs in self.observers:
                continue
                vision = self.race.render_data[obs].image
                mask = get_mask(self.race.render_data[obs].instance)

                coor = abscoor2imagecoor(
                    np.array(self.ws.players[self.obs_player].camera.projection),
                    np.array(self.ws.players[self.obs_player].camera.view),
                    np.array(self.ws.soccer.ball.location)
                )
                # coor = self.mask2coor(mask)
                
                if self.record_video:
                    video_vision = vision
                    if self.draw:
                        drawed_vision = self.to_draw(vision, all_actions)
                        to_video.append(drawed_vision)

                if self.record_td and self.inclusion_criteria(coor, mask):
                    to_td.append((vision, coor, mask))

            # Logs
            if i % self.print_each == 0:
                print('Frame: ', i)
                if self.record_td:
                  print('Len td: ', len(to_td))
            
            # if time1 > max_time1:
            #     max_time1 = time1
            #     print('Team1')
            #     print('New Max: ', max_time1)

            # if time2 > max_time2:
            #     max_time2 = time2
            #     print('Team2')
            #     print('New Max: ', max_time2)

        if self.record_video:
            save_video(to_video)
        if self.record_td:
            print('Len td', len(to_td))
            save_td(to_td)

    def get_actions(self, team):
        from time import time
        th = self.team1.num_players
        get_team_part = (lambda ls: ls[:th]) if team.team == 0 else (lambda ls: ls[th:])
        get_other_part = (lambda ls: ls[th:]) if team.team == 0 else (lambda ls: ls[:th])

        team_players = get_team_part(self.ws.players)
        team_states = [player2state(player) for player in team_players]
        other_players = get_other_part(self.ws.players)
        other_states = [player2state(player) for player in other_players]
        soccer_state = soccerobject2state(self.ws.soccer)
        team_vision = [el.image for el in get_team_part(self.race.render_data)]

        elapsed = time()
        if type(team).agent_type == 'image':
            team_actions = team.act(team_states, team_vision)
        else:
            team_actions = team.act(team_states, other_states, soccer_state)
        
        elapsed = time() - elapsed
        team_actions = [action_packer(action_dict) for action_dict in team_actions]
        return team_actions, elapsed
        

    def to_draw(self, vision: np.ndarray, all_actions) -> np.ndarray:
        logs = {}
        draws_circles = []

        if self.draw_info_t1:
            t1_logs, t1_centers = self.race_manager.team1.get_draw_info()
            logs.update(t1_logs)
            draws_circles += t1_centers

        if self.draw_info_t2:
            t2_logs, t2_centers = self.race_manager.team2.get_draw_info()
            logs.update(t2_logs)
            draws_circles += t2_centers
        
        drawn = self.drawer.draw_logs(vision, logs, 'green')

        for center, color in draws_circles:
            drawn = self.drawer.draw_circle(drawn, center, color)
        drawn = self.drawer.draw_axis(drawn)
          
        return drawn

    def end_game(self):
        self.race_manager.race.stop()
        del self.race_manager


if __name__ == '__main__':
    parser = ArgumentParser("Collects a dataset for the high-level planner")
    parser.add_argument('-o', '--output', default=DATASET_PATH)
    parser.add_argument('-n', '--n_frames', default=10000, type=int)
    parser.add_argument('team1', help="Python module name or `AI` for AI players.", nargs='?')
    parser.add_argument('team2', help="Python module name or `AI` for AI players.", nargs='?')
    parser.add_argument('-width', type=int, default=400, help="Width of the vision")
    parser.add_argument('-height', type=int, default=600, help="Width of the height")
    
    parser.add_argument('-vid', '--record_video', action='store_true', default=False, help="Do you want to record a video?")
    parser.add_argument('-td', '--generate_td', action='store_true', default=False, help="Do you want to collect samples?")
    parser.add_argument('-draw', '--draw', action='store_true', default=False, help="Do you want to collect samples?")

    args = parser.parse_args()

    race_manager = None
    match = None

    resolution = (args.width, args.height)

    team1 = TEAM_DICT[args.team1]()
    team2 = TEAM_DICT[args.team2]()

    team1.new_match(team=0, num_players=2)
    team2.new_match(team=1, num_players=2)

    try:
        race_manager = RacerManager(team1, team2, resolution)
        match = Match(race_manager, args.record_video, args.draw, args.generate_td, args.n_frames)
        match.play()
    except Exception as e:
        print("Something went wrong !!")
        print(traceback.format_exc())
    finally:
        if match:
            print("Match finished.")
            match.end_game()
    
