import numpy as np
from PIL import Image

class DroneSimulator:
    """
        The DroneSimulator class should provide a good way to train some
        reinforcement learning algorithms in case of a flock of drones that
        must reach some target. They can communicate each other thanks the
        stigmation space.

        Parameters
        ----------
        bitmap : str
            This must be the path of a bitmap that must rappresent the inital
            space, this must contain at least three different color:
                - one for ground (must be full black, no bit at 1)
                - one for obstacles
                - one for targets
            Colours for items must follow this rule:
                they can have ONLY one bit at 1 and others must be a 0.
                So, at except for the ground, valid colours might be 1000...,
                01000..., 00100....
                This is useful for rappresenting overlap beetween items,
                for example the colour: 110000... rappresent the overlapping
                between a target and a obstacle.
                Colours will be ordered in level by the position of the bit at
                1, the most significant bit will be the first level, and so on
                So, a bitmap can be rappresent by:
                    - 0000... (no bit at 1):                ground
                    - 1000... (only first bit at 1):        obstacles
                    - 0001... (only second bit at 1):       targets
                    - Other colours will rappresent other infos

        batch_size: int
            The number of parallel simulation to run.
            The render will be allow only with batch_size equal to 1.

        observation_range: int
            The observation range will be a square with a length of the side
            equal to the provided observation_range.

        amount_of_drone: int
            The amount of drone that will be random positioned in the
            environment.

        stigmation_evaporation_speed: np.array(N)
            This must be a np.array with one dimension.
            Each element of the array rappresent the evaporation speed for the
            corresponding stigmergy level.
            The length of the array (N) rappresent the number of stigmergy
            levels.

        reward_function: function
            This must be a reference to reward_function and must accept one
            parameter (np.array) that rappresent the full environment with
            obstacles, targets, stigmergy space, drones.

        max_steps: int
            This integer rappresent the maximum number of step.
            So the step function return done equal to true when this number of
            step will be reached, unless the targets have already been achieved.

        Attributes
        ----------
        __batch_size: int
            This is where we store batch_size.

        __observation_range: int
            This is where we store observation_range.

        __amount_of_drone: int
            This is where we store amount_of_drone.

        __stigmation_evaporation_speed: np.array(N)
            This is where we store stigmation_evaporation_speed.

        __reward_function: function
            This is where we store reward_function.

        __max_steps: int
            This is where we store max_steps.

        __env: np.array
            This rappresent the full environment, the level in this array will
            be ordered in this way:
            - Drones: will be marked with an int to see the different drones
            - Obstacles: the position of obstacles
            - Target: the position of targets
            - Other items imported from the initial bitmap
            - Stigmergy level in according to the stigmation_evaporation_speed

        TODO: add other attributes

    """
    def __init__(self,
        bitmap,
        batch_size,
        observation_range,
        amount_of_drone,
        stigmation_evaporation_speed,
        reward_function,
        max_steps
        ):
        self.__batch_size = batch_size
        self.__observation_range = observation_range
        self.__amount_of_drone = amount_of_drone
        self.__stigmation_evaporation_speed = stigmation_evaporation_speed
        self.__reward_function = reward_function
        self.__max_steps = max_steps

        self.__env = np.asarray(Image.open(bitmap))
        print(bitmap_array[bitmap_array > 0])
        last_level = None
        while


    def step():
        raise NotImplementedError

    def render():
        raise NotImplementedError

    def __bitmap_to_tensor(bitmap):
