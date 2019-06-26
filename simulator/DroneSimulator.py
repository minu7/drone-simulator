import numpy as np
from PIL import Image
import sys

np.set_printoptions(threshold=sys.maxsize)

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

        __stigmation_evaporation_speed: np.ndarray(N)
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

        env = self.__bitmap_to_tensor(bitmap)

        stigmergy_space = np.zeros((
                self.__stigmation_evaporation_speed.shape[0],
                env.shape[1],
                env.shape[2]
            ))
        env = np.vstack((env, stigmergy_space))


        self.__env = self.__add_drones_in_batch(env, batch_size, amount_of_drone)


    def step(actions):
        """
        This methods allow to take actions for all drones and in all
        batch dimensions.

        Parameters
        ----------
        actions : np.array[batch_size, amount_of_drones, action_dimension]
            The actions must represent all the action taken by the drones in all
            dimension, the order of the drones must be in according to the id of
            the drones, so a drone with id 1 must be the first action.
            action_dimension: 4 + stigmergy_level * 2
                - First 4 are cardinal dimension, they represent the movement,
                  so es [1, 0, 0, 0], this drone will move to the top of the
                  screen only 0 and 1 are allowed for the dimension.
                - Other levels will be the radius and intensity for each action
                  in the stigmergy level, so es: [0, 0] no action in the level
                  will be taken, [5, 3]: in this stigmergy level the drone
                  leaves 3 in a radius of 5 around itself

        Raises
        ------
        RuntimeError


        Returns
        -------
        (observation, reward, done)
            observation :
                np.array[batch_size, amount_of_drones, observation_range]
                    this will be a slice of the full environment in the
                    observation_range around the each drone.
            reward : np.array
                the reward function will be ran in each batch dimension with
                input the full environment, the array of result will be returned
                here.
            done : bool
                will be true if the max_steps is reached or all targets will be 
                achieved by drones, else will be false

        """
        raise NotImplementedError

    def render():
        raise NotImplementedError


    def __bitmap_to_tensor(self, bitmap):
        """
        This methods converts a bitmap to a tensor according the representation
        written above in the constructor, so we need at least two colours with
        only one bit at 1 for obstacles and targets.

        Parameters
        ----------
        bitmap : str
            The path of the input bitmap

        Raises
        ------
        RuntimeError
            IOError: The path of the bitmap is invalid

        Returns
        -------
        np.ndarray
            the tensor obtained from bitmap
        """
        input_array = np.asarray(Image.open(bitmap))
        rgb_bit_array = np.unpackbits(input_array, axis=2)
        # here i have rgb_bit_array that is an array 2d with all cells are an
        # array of bit
        tensor = []
        for i in range(0, 24):
            level = rgb_bit_array[:, :, i]
            if np.any(level):
                # only level with at least 1 item are inserted in env
                tensor.append(level)

        if len(tensor) < 2:
            raise Exception("At least obstacles and targets must be provided")
        return np.asarray(tensor)


    def __add_drones_in_batch(self, env, batch_size, amount_of_drone):
        """
        This methods adds drones and the batch_size dimension.

        Parameters
        ----------
        env : np.ndarray
            The environment without batch dimension and the drones
        batch_size : int
            The batch_size represent how big must be the batch dimension

        Raises
        ------
        RuntimeError


        Returns
        -------
        np.ndarray
            the full environment
        """
        # for the creation of drone_level i create an array with the dimension
        # of the other levels with only zeros, I add many ones as many as the
        # number of drones, i reshape the result to the corresponding dimension,
        # then the result will be shuffled
        drone_array = np.zeros(env.shape[1] * env.shape[2])
        drone_array[:amount_of_drone] = np.arange(amount_of_drone)
        np.random.shuffle(drone_array)
        drone_level = np.reshape(drone_array, env.shape[1:3])
        drone_level = drone_level[np.newaxis, ...]
        env = np.vstack((drone_level, env))
        # adding batch dimension
        env = env[np.newaxis, ...]
        env = np.repeat(env, batch_size, axis=0)

        return env
