import sys
import random

import numpy as np
from PIL import Image

np.set_printoptions(threshold=sys.maxsize)

class DroneSimulator:
    """
        The DroneSimulator class should provide a good way to train some
        reinforcement learning algorithms in case of a flock of drones that
        must reach some target. They can communicate each other thanks the
        stigmation space.

        Parameters
        ----------
        bitmap: str
            This must be the path of a bitmap that must rappresent the inital
            space, this must contain at least three different color:
                - one for ground (must be full black, no bit at 1)
                - one for targets
                - one for obstacles
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
                    - 1000... (only first bit at 1):        targets
                    - 0001... (only second bit at 1):       obstacles
                    - Other colours will rappresent other infos

        batch_size: int
            The number of parallel simulation to run.
            The render will be allow only with batch_size equal to 1.

        observation_range: int
            The observation range will be a square with a length of the side
            equal to the provided observation_range.

        amount_of_drones: int
            The amount of drone that will be random positioned in the
            environment.

        stigmation_evaporation_speed: np.array(N)
            This must be a np.array with one dimension.
            Each element of the array rappresent the evaporation speed for the
            corresponding stigmergy level.
            The length of the array (N) rappresent the number of stigmergy
            levels.

        inertia: float
            This serves to have an inertia in the movement of the drones.
            When a command is given by drone the new position will be:
            position = position + velocity * t
            where velocity = velocity * inertia + command * ( 1 - inertia)

        collision_detection: np.ndarray(M)
            When a bitmap is imported beyond the first level of targets other
            levels can be imported and this array must represent in which other
            levels the collision with drones must be detected

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

        __amount_of_drones: int
            This is where we store amount_of_drones.

        __stigmation_evaporation_speed: np.ndarray(N)
            This is where we store stigmation_evaporation_speed.

        __inertia: float
            This is where we store inertia.

        __reward_function: function
            This is where we store reward_function.

        __max_steps: int
            This is where we store max_steps.

        __env: np.array
            This rappresent the environment with collision

        TODO: add other attributes

    """
    def __init__(self,
        bitmap,
        batch_size,
        observation_range,
        amount_of_drones,
        stigmation_evaporation_speed,
        inertia,
        collision_detection,
        reward_function,
        max_steps
        ):
        self.__batch_size = batch_size
        self.__observation_range = observation_range
        self.__amount_of_drones = amount_of_drones
        self.__stigmation_evaporation_speed = stigmation_evaporation_speed
        self.__inertia = inertia
        self.__reward_function = reward_function
        self.__max_steps = max_steps

        # env and collision are divided only for render purpose
        self.__env = np.array([])
        self.__no_collision = np.array([])
        self.__targets = np.array([])

        self.__parse_bitmap(bitmap, collision_detection)

        # The collision will be detected only with this matrix because contain
        # all info necessary
        self.__collision = np.sum(self.__env, axis = 0)
        self.__collision[self.__collision > 0] = 1

        # Initial position equal to -1 is for see which drones must be
        # positioned yet
        self.__drones_position = np.full((amount_of_drones, 2), -1)
        self.__drones_velocity = np.zeros((amount_of_drones, 2))
        self.__drawn_drones = np.array([])

        self.__stigmergy_space = np.zeros((
                self.__stigmation_evaporation_speed.shape[0],
                self.__targets.shape[0],
                self.__targets.shape[1]
            ))

        # Only for stigmergy_space and drones the batch dimension will be added
        # because is the only 2 levels that can change beetween batch the other
        # will be fixed

        self.__stigmergy_space = self.__add_batch_dimension(
            self.__stigmergy_space
        )
        self.__drones_velocity = self.__add_batch_dimension(
            self.__drones_velocity
        )
        self.__drones_position = self.__add_batch_dimension(
            self.__drones_position
        )

        self.__init_drones()


    def step(actions):
        """
        This methods allow to take actions for all drones and in all
        batch dimensions.

        Parameters
        ----------
        actions: np.array[batch_size, amount_of_drones, action_dimension]
            The actions must represent all the action taken by the drones in all
            dimension, the order of the drones must be in according to the id of
            the drones, so a drone with id 1 must be the first action.
            action_dimension: 2 + stigmergy_level * 2
                - First 4 are cardinal dimension, they represent the movement,
                  so es [1, 0], this drone will move to the top of the
                  screen only 0 and 1 are allowed for the dimension.
                  (for the bottom [-1, 0])
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
            observation:
                np.array[batch_size, amount_of_drones, observation_range]
                    this will be a slice of the full environment in the
                    observation_range around the each drone.
            reward: np.array
                the reward function will be ran in each batch dimension with
                input the full environment, the array of result will be returned
                here.
            done: bool
                will be true if the max_steps is reached or all targets will be
                achieved by drones, else will be false

        """
        raise NotImplementedError

    def render():
        raise NotImplementedError


    def __parse_bitmap(self, bitmap, collision_detection):
        """
        This methods converts a bitmap to a tensor according the representation
        written above in the constructor, so we need at least two colours with
        only one bit at 1 for obstacles and targets.

        Parameters
        ----------
        bitmap: str
            The path of the input bitmap

        collision_detection: np.array
            Array of booleans that represent with which levels the collision
            must be checked

        Raises
        ------
        RuntimeError
            IOError: The path of the bitmap is invalid

        Returns
        -------
        This method has side effect to the object, so it will change:
            - self.__env:
                Batch dimension will be added
            - self.__drones_position:
                Drones will be created and positioned in batch dimension
            - self.__drones_velocity:
                The initial velocity of all drones will be zero
            - self.__stigmergy_space:
                Batch dimension will be added
            - self.__items_without_collition:
                Batch dimension will be added
        """
        input_array = np.asarray(Image.open(bitmap))
        rgb_bit_array = np.unpackbits(input_array, axis=2)
        # here i have rgb_bit_array that is an array 2d with all cells are an
        # array of bit
        env = []
        no_collision = []
        level_founded = 0
        for i in range(0, 24):
            level = rgb_bit_array[:, :, i]
            if np.any(level):
                # only level with at least 1 item are inserted in the
                # environment
                if level_founded == 0:
                    # first level are targets
                    self.__targets = np.asarray(level)
                else:
                    if collision_detection[level_founded - 1]:
                        env.append(level)
                    else:
                        no_collision.append(level)
                level_founded += 1

        self.__env = np.asarray(env)
        self.__no_collision = np.asarray(no_collision)


    def __init_drones(self):
        """
        This methods return drones and the batch_size dimension.

        Parameters
        ----------

        Raises
        ------
        RuntimeError


        Returns
        -------
        This method has side effect to the object, so it will change:
            - self.__drones_position:
                Drones will be created and positioned in batch dimension
        """
        position = np.asarray([
            random.randint(0, self.__targets.shape[0] - 1),
            random.randint(0, self.__targets.shape[1] - 1)
        ])



    def __render(self):
        """
        This method create a level for each drones and draw drones one for
        level.
        This representation make easy check collision and plot the actual
        situation of the environment.

        Parameters
        ----------
        batch: int
            only the provided batch will be rendered, if None all batch will be
            rendered

        Raises
        ------
        RuntimeError

        Returns
        -------

        This method has side effect to the object, so it will change:
            - self.__drawn_drones:
                The drones will be rendered one for level

        """
        raise NotImplementedError


    def __detect_collision(self):
        """
        This method return true there is a collision in this actual situation.

        Parameters
        ----------

        Raises
        ------
        RuntimeError

        Returns
        -------
        bool

        """
        raise NotImplementedError

    def __drawn_drone(self, positionVelocity):
        """
        This method return a level with a drone rendered in the provided
        position.
        This function must be called after the parsing of bitmap and init the
        batch dimension

        Parameters
        ----------
            positionVelocity: np.array(4)
                first two elements of array represent position, other two are
                for velocity

        Raises
        ------
        RuntimeError

        Returns
        -------
        np.array()
            Represent the level with the drones

        """
        level = self.__stigmergy_space = np.zeros((
                self.__targets.shape[0],
                self.__targets.shape[1]
            ))
        # This is for uninitialized drones
        if position[0] < 0 or position[1] < 0:
            return level
        # The drone for now it's a simple square
        # velocity is provided for future better representation of drone
        # TIPS: the bigger margin is + 2 because is not included in the slice
        level[position[0] - 1 : position[0] + 2,
            position[1] - 1 : position[1] + 2] = 1
        return level

    def __add_batch_dimension(self, matrix):
        """
        This function add batch dimension and repeat the inital matrix on this
        axis.

        Parameters
        ----------
            matrix: np.array
                initial matrix

        Raises
        ------
        RuntimeError

        Returns
        -------
        np.array()
            Represent the new matrix

        """
        matrix = matrix[np.newaxis, ...]
        matrix = np.repeat(matrix, self.__batch_size, axis=0)
        return matrix
