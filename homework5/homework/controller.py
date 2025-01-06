import pystk
import numpy as np


def control(aim_point, current_vel):
    """
    Set the Action for the low-level controller
    :param aim_point: Aim point, in screen coordinate frame [-1..1]
    :param current_vel: Current velocity of the kart
    :return: a pystk.Action (set acceleration, brake, steer, drift)
    """
    action = pystk.Action()
    acceleration = 0
    brake = False
    steer = 0
    drift = False
    nitro = False

    h = 96
    w = 128
    a = 0
    b = 0.125

    # Steer
    steer = -np.arctan(((aim_point[0]-a)*w) / ((aim_point[1]-b)*h))*(2/np.pi)
    

    # target_velocity = 40
    target_velocity = 55/abs(steer)**0.08
    

    # Drift and Nitro
    drift_limit = 0.5
    if drift_limit < abs(steer):
        drift = True
    
    nitro_limit = 0.2
    if abs(steer) < nitro_limit:
        nitro = True
    
    # Brake
    brake_limit = 10
    if current_vel - target_velocity > brake_limit or abs(steer)>0.4:
        brake = True
    
    # Acceleration(cv, tv, steer)
    # acceleration_limit = 20
    if current_vel < target_velocity:# and abs(steer)<0.45:
        acceleration = (target_velocity - current_vel) / target_velocity #*(1-abs(steer))**2
    elif current_vel < 5:
        acceleration = (target_velocity - current_vel) / target_velocity 
 
    steer *= 1.25
    action.acceleration = acceleration
    action.steer = steer
    action.drift = drift
    action.brake = brake
    action.nitro = nitro

    return action

    """
    Your code here
    Hint: Use action.acceleration (0..1) to change the velocity. Try targeting a target_velocity (e.g. 20).
    Hint: Use action.brake to True/False to brake (optionally)
    Hint: Use action.steer to turn the kart towards the aim_point, clip the steer angle to -1..1
    Hint: You may want to use action.drift=True for wide turns (it will turn faster)
    """


if __name__ == '__main__':
    from .utils import PyTux
    from argparse import ArgumentParser

    def test_controller(args):
        import numpy as np
        pytux = PyTux()
        for t in args.track:
            steps, how_far = pytux.rollout(t, control, max_frames=args.msteps, verbose=args.verbose)
            print(steps, how_far)
        pytux.close()


    parser = ArgumentParser()
    parser.add_argument('track', nargs='+')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-msteps', default=1000, type=int)
    args = parser.parse_args()
    test_controller(args)