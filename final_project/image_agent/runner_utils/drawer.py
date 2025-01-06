import imgviz
import numpy as np
from typing import Tuple


class Drawer(object):
    def __init__(self):
        self.diameter = 50
        self.width = 5
        self.start_corner = (10, 10)
        self.color_map = {
            'black' : (0, 0, 0),
            'red' : (300, 0, 0),
            'green' : (0, 300, 0),
            'blue' : (0, 0, 300),
            'white' : (300, 300, 300),
        }
        self.letter_size = 10
 
    def draw_circle(self, vision: np.ndarray, center: Tuple[int], color: str) -> np.ndarray:
        drawn = np.array(vision)
        color = self.color_map[color]
        drawn = imgviz.draw.circle(
            drawn, center, self.diameter, None, color, self.width
        )

        return drawn

    def draw_logs(self, vision, logs: dict, color: str) -> np.ndarray:
        drawn = np.array(vision)
        color = self.color_map['black']

        text = ''
        for key, value in logs.items():
            text += f'{key} : {value} \n'

        drawn = imgviz.draw.text(
            drawn, self.start_corner, text, self.letter_size, color
        )

        return drawn
    
    def draw_axis(self, vision) -> np.ndarray:
        shape = np.array(vision.shape)[0:2]
        y_tot, x_tot = shape
        y_med, x_med = shape // 2

        color = self.color_map['white']
        drawn = imgviz.draw.rectangle(
            vision, (y_med, 0), (y_med, x_tot), color
        )
        drawn = imgviz.draw.rectangle(
            drawn, (0, x_med), (y_tot, x_med), color
        )
        return drawn