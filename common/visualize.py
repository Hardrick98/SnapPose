import typing as tp

import numpy as np

human_36m_joints = {
    0: 'root',
    1: 'rhip',
    2: 'rkne',
    3: 'rank',
    4: 'lhip',
    5: 'lkne',
    6: 'lank',
    7: 'belly',
    8: 'neck',
    9: 'nose',
    10: 'head',
    11: 'lsho',
    12: 'lelb',
    13: 'lwri',
    14: 'rsho',
    15: 'relb',
    16: 'rwri'
}

chains_ixs = (
    [3, 2, 1, 0, 4, 5, 6],
    [0, 7, 8, 9, 10],
    [13, 12, 11, 8, 14, 15, 16]
)

def get_chain_dots(dots: np.ndarray, chain_dots_indexes: tp.List[int]) -> np.ndarray:  # chain of dots
    return dots[chain_dots_indexes]

def get_chains(dots: np.ndarray, legs_chain_ixs: tp.List[int], torso_chain_ixs: tp.List[int], arms_chain_ixs: tp.List[int]):
    return (get_chain_dots(dots, legs_chain_ixs),
            get_chain_dots(dots, torso_chain_ixs),
            get_chain_dots(dots, arms_chain_ixs))

def subplot_nodes(dots: np.ndarray, ax, c='red'):
    return ax.scatter3D(dots[:, 0], dots[:, 2], dots[:, 1], c=c)

def subplot_bones(chains: tp.Tuple[np.ndarray, ...], ax, c='greens'):
    return [ax.plot(chain[:, 0], chain[:, 2], chain[:, 1], c=c) for chain in chains]