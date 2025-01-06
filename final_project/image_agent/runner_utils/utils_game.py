import pystk

def player2state(player: pystk.Player) -> dict:
    camera_info = {
        'aspect': player.camera.aspect,
        'fov': player.camera.fov,
        'mode': player.camera.mode,
        'projection': player.camera.projection,
        'view': player.camera.view,
    }

    kart_info = {
        'front': player.kart.front,
        'location': player.kart.location,
        'rotation': player.kart.rotation,
        'size': player.kart.size,
        'velocity': player.kart.velocity,
    }
    
    return {
        'camera' : camera_info,
        'kart' : kart_info
    }

def action_packer(action_dict) -> pystk.Action:
    action = pystk.Action()
    for name, value in action_dict.items():
        setattr(action, name, value)
    return action


def soccerobject2state(soccer: pystk.Soccer) -> dict:
    return {
        'ball' : {
            'id' : soccer.ball.id,
            'location' : soccer.ball.location,
            'size' : soccer.ball.size,
        },
        'goal_line' : soccer.goal_line,
        'score' : soccer.score,
    }