from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gym import spaces
from .act_space import ActTuple


def back_to_act_tuple(inputs):
    """
    Args
    ----
    inputs : shape=[num_agents, 11, 2]
        - off_action_space shape=(6, 2)  # [decision, ball_dir] + player_dash(5, 2)
        - def_action_space shape=(5, 2)  # player_dash(5, 2)

    Returns
    -------
    Tuple(Discrete(3), Box(), Box(5, 2), Box(5, 2))
    """
    transformed = []
    for input_ in inputs:
        transformed.append(
            np.array([
                input_[0, 0],  # Discrete(3)
                input_[0, 1],  # Box()
                input_[1:6, :],  # Box(5, 2)
                input_[6:11, :]  # Box(5, 2)
            ])
        )
    return transformed
