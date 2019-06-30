from simulator.DroneSimulator import DroneSimulator
import numpy as np

drone_simulator = DroneSimulator(bitmap = './test-with-2-levels.bmp',
    batch_size = 1,
    observation_range = 2,
    amount_of_drones = 4,
    stigmation_evaporation_speed = np.array([1, 2, 3]),
    collision_detection = np.array([True, False]), # with the first level after targets
    # a collision must be detected
    inertia = 0.8,
    reward_function = 5,
    max_steps = 6,
    render = True)

drone_simulator.render()
