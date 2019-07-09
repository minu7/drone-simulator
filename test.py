from simulator.DroneSimulator import DroneSimulator
import numpy as np

drone_simulator = DroneSimulator(bitmap = './test-very-little.bmp',
    batch_size = 1,
    observation_range = 2,
    amount_of_drones = 3,
    stigmation_evaporation_speed = np.array([1, 2, 3]),
    collision_detection = np.array([True, True, True, True]),
    inertia = 0.3,
    reward_function = 5,
    max_steps = 6,
    render = True)

drone_simulator.render()

# first batch dimension, drone_dimension, action_dimension
drone_simulator.step(np.ones((1, 1, 4)))
