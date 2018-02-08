from gym import Space
import numpy as np


class ActTuple(Space):
    """
    A tuple (i.e., product) of simpler spaces

    Example usage:
    self.observation_space = spaces.Tuple((spaces.Discrete(2), spaces.Discrete(3)))
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
        return np.float32