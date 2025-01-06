import pystk
import numpy as np

from skimage import draw


class CircleDrawer:
    pass

class RacerManager(object):
    def __init__(self):
        graphics_config = self.do_graphics_config()
        pystk.init(graphics_config)

        race_config = self.do_race_config()
        self.race = pystk.Race(race_config)
        self.world_state = pystk.WorldState()

        self.max_frames = 600

        self.do_td = True
        self.do_video = True
        self.draw_circle_control = True

        self.obs_player = 0
    
    def do_graphics_config(self):
        graphics_config = pystk.GraphicsConfig.hd()
        graphics_config.screen_width = 800
        graphics_config.screen_height = 600
        return graphics_config

    def do_race_config(self):
        race_config = pystk.RaceConfig()

        race_config.num_kart = 2
        for i in range(race_config.num_kart):
            race_config.players.append(
                pystk.PlayerConfig(
                    controller = pystk.PlayerConfig.Controller.AI_CONTROL,
                    team = 0
                )
            )
        race_config.mode = pystk.RaceConfig.RaceMode.SOCCER
        race_config.track = 'icy_soccer_field'

        return race_config

    def run(self):
        self.race.start()

        to_video = []
        to_td = []

        for i in range(self.max_frames):
            self.race.step()
            self.world_state.update()

            vision = self.race.render_data[self.obs_player].image
            mask = self.get_mask()

            # coor = self.get_puck_coor()
            coor = self.mask2coor(mask)

            vision = self.draw_circle(vision, coor)
            

            if self.do_video:
                video_vision = vision
                # if self.draw_circle_control:
                #     video_vision = self.draw_circle(vision, coor)
                to_video.append(video_vision)
            
            if self.do_td:
                to_td.append((vision, coor, mask))
            
            if i % 100 == 0:
                print(i)

        self.save_video(to_video)
        self.save_td(to_td)
    
    def get_mask(self):
        puck_object_type = 8
        instances_code = (self.race.render_data[self.obs_player].instance >> pystk.object_type_shift)
        return instances_code == puck_object_type
    
    def get_puck_coor(self):
        puck_position = self.world_state.soccer.ball.location
        proj_matrix = np.array(self.world_state.players[self.obs_player].camera.projection).T
        view_matrix = np.array(self.world_state.players[self.obs_player].camera.view).T

        puck_coor = proj_matrix @ view_matrix @ np.array(list(puck_position) + [1])
        puck_coor = np.clip(np.array([puck_coor[0] / puck_coor[-1], -puck_coor[1] / puck_coor[-1]]), -1, 1)
        
        return puck_coor
    
    def mask2coor(self, mask):
        sx = 0
        sy = 0
        puk_count = 0

        i = 0
        for y in mask:
          j = 0
          for x in y:
            if x == 1:
              sx = sx + i
              sy = sy + j
              puk_count += 1
            j += 1
          i += 1
        
        if not puk_count:
            return np.array([-1, -1])

        sx = sx / puk_count
        sy = sy / puk_count

        a = np.array([sx, sy])
        return (a + 1 - (np.array(mask.shape) / 2)) / (np.array(mask.shape) / 2)

    # def coords2heatmap(self, coord):
    #     screen_width = 800
    #     screen_height = 600
    #     shape = (screen_height, screen_width)
    #     heatmap = np.zeros(shape)
    #     a = np.array([screen_width, screen_height])
    #     coords = np.floor((ball_on_cam + 1)/2 * (a-1))
    #     heatmap[a] = 1
    #     return heatmap

    def draw_circle(self, vision, coor_puck):
        drawed_vision = np.array(vision)
        screen_shape = np.array(vision.shape[:2])

        indices = (coor_puck + 1) / 2 * (screen_shape - 1)
        indices = np.floor(indices)

        rr, cc = draw.circle(*indices, 40, shape=screen_shape)
        drawed_vision[rr, cc] = np.array([256, 0, 0])

        return drawed_vision


    def save_video(self, frames, fps=30):
        # Frames should be an iterable of ndarray
        import imageio
        imageio.mimsave('avideo.mp4', frames, fps=fps, bitrate=10**7)
    
    def save_td(self, entries):
        # coors = [i[1] for i in entries]
        # for a in coors[30:600:10]:
        #   print(a)

        counter = 0
        with_count = lambda s: s + '_' + str(counter)

        for vision, coor, mask in entries:
            # Save vision
            from PIL import Image
            vision_fn = with_count('vision') + '.png'
            Image.fromarray(vision).save(vision_fn)

            # Save coor
            coor_fn = with_count('coor') + '.csv'
            with open(coor_fn, 'w') as f:
                f.write('%f,%f' % tuple(coor))
            
            # Save mask
            mask_fn = with_count('mask') + '.png'
            Image.fromarray(mask).save(mask_fn)

            counter += 1

if __name__ == '__main__':
    a = RacerManager()
    a.run()