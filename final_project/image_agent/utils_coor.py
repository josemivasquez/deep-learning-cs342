import numpy as np

def imagecoor2indices(coor, shape):
    return np.clip(
        np.floor((np.flip(coor) + 1) / 2 * shape), 
        np.array([0, 0]), shape
    )

def indices2imagecoor(indices, shape):
    return np.flip(((indices / shape) * 2) - 1)

def abscoor2imagecoor(proj: np.array, view: np.array, abs_coor: np.array):
    proj = proj.T
    view = view.T

    response_coor = proj @ view @ np.array(list(abs_coor) + [1])
    response_coor = np.clip(np.array(
        [response_coor[0] / response_coor[-1], -response_coor[1] / response_coor[-1]]
        ), -1, 1)
    
    return response_coor

def imagecoor2abscoor(coor, proj, view) -> np.ndarray:
    proj = proj.T
    view = view.T
    m = proj @ view
    h = 0.3698
    
    coefs = np.array([
        [m[0][0], m[0][2], -coor[0], 0],
        [m[1][0], m[1][2], coor[1], 0],
        [m[2][0], m[2][2], 0, -1],
        [m[3][0], m[3][2], -1, 0]
    ])
    rigth = np.array(
        [-m[0][3]-m[0][1]*h, -m[1][3]-m[1][1]*h, -m[2][3]-m[2][1]*h, -m[3][3]-m[3][1]*h]
    )
    
    l = np.linalg.inv(coefs)
    sol = l @ rigth

    x, y = sol[0], sol[1]
    abscoor = np.array([x, h, y])

    return abscoor

def get2d(v):
    return np.array([v[0], v[2]])
