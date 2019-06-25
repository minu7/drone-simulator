from simulator.DroneSimulator import DroneSimulator
import numpy as np

DroneSimulator(bitmap = './test2.bmp',
    batch_size = 1,
    observation_range = 2,
    amount_of_drone = 3,
    stigmation_evaporation_speed = np.array([1, 2, 3]),
    reward_function = 5,
    max_steps = 6)
