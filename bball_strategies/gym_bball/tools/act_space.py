from gym import Space
import tensorflow as tf


class ActTuple(Space):
    """ customized action scace containing one Discrete() and three continuous Box()
    """

    def __init__(self, spaces):
        self.spaces = spaces

    def sample(self):
        return tuple([space.sample() for space in self.spaces])

    def contains(self, x):
        if isinstance(x, list):
            x = tuple(x)  # Promote list to tuple for contains check
        return isinstance(x, tuple) and len(x) == len(self.spaces) and all(
            space.contains(part) for (space, part) in zip(self.spaces, x))

    def __repr__(self):
        return "Tuple(" + ", ". join([str(s) for s in self.spaces]) + ")"

    def to_jsonable(self, sample_n):
        # serialize as list-repr of tuple of vectors
        return [space.to_jsonable([sample[i] for sample in sample_n])
                for i, space in enumerate(self.spaces)]

    def from_jsonable(self, sample_n):
        return zip(*[space.from_jsonable(sample_n[i]) for i, space in enumerate(self.spaces)])

    # Extended, this function is additional to gym.spaces.tuple_space
    def __eq__(self, other):
        return all([space == o_space for space, o_space in zip(self.spaces, other.spaces)])

    def __iter__(self):
        for v in list(self.spaces):
            yield v

    def __getitem__(self, idx):
        return self.spaces[idx]

    # Extended
    @property
    def shape(self):
        """ 
        ### original shape
        - Tuple(Discrete(3), Box(), Box(5, 2), Box(5, 2))

        ### reshape to (11, 2)
        - spaces[0,0] = Discrete(3), 
        - spaces[0,1] = Box(), 
        - spaces[1:6,:] = Box(5, 2), 
        - spaces[6:11,:] = Box(5, 2) 
        """
        return (11, 2)
    # Extended

    @property
    def dtype(self):
        return tf.float32


def back_to_act_tuple(inputs):
    """
    Args
    ----
    inputs : shape=[num_agents, 11, 2]
        - off_action_space shape=(6, 2)  # [decision, ball_dir] + player_dash(5, 2)
        - def_action_space shape=(5, 2)  # player_dash(5, 2)

    Returns
    -------
    [num_agents, [Discrete(3), Box(), Box(5, 2), Box(5, 2)]]
    """
    transformed = []
    for input_ in inputs:
        transformed.append(
            [
                int(round(input_[0, 0])),  # Discrete(3) must be int
                input_[0, 1],  # Box()
                input_[1:6, :],  # Box(5, 2)
                input_[6:11, :]  # Box(5, 2)
            ]
        )
    return transformed
