from simulator.DroneSimulator import DroneSimulator
import numpy as np

DroneSimulator(bitmap = './test2.bmp',
    batch_size = 2,
    observation_range = 2,
    amount_of_drones = 10,
    stigmation_evaporation_speed = np.array([1, 2, 3]),
    collision_detection = np.array([True]), # with the first level after targets
    # a collision must be detected
    inertia = 0.8,
    reward_function = 5,
    max_steps = 6)
