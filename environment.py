import numpy as np


class Environment(object):
    def __init__(self, n_levels, n_lights, n_lights_tuple):
        self.n_levels = n_levels
        self.n_lights = n_lights
        self.n_lights_tuple = n_lights_tuple
        self.state = np.zeros((self.n_levels, self.n_lights)).astype(np.int)
        self.high_lev_change = np.zeros((self.n_levels-1, self.n_lights)).astype(np.int)

    def respond(self, action):
        switch_to, light_i = action
        self.state[0, light_i] = switch_to
        return self.do_events(light_i)

    def do_events(self, light_i):
        self.high_lev_change[:, :] = 0
        for level in range(self.n_levels - 1):  # check for each level if a tuple is full
            n_lights_tuple = self.n_lights_tuple ** (1 + level)
            first_in_tuple = light_i - (light_i % n_lights_tuple)
            tuple = range(first_in_tuple, first_in_tuple + n_lights_tuple)
            tuple_complete = np.all(self.state[level, tuple])  # check if all lights in a tuple are on
            next_level_free = not np.any(self.state[level + 1, tuple])  # check if next-level lights are still off
            if tuple_complete and next_level_free:
                self.state[level, tuple] = 0  # turn off lower-level lights
                self.state[level + 1, tuple] = 1  # turn on next-level lights
                self.high_lev_change[level, tuple] = 1
