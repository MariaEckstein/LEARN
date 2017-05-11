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
        first_in_tuple = light_i - (light_i % self.n_lights_tuple)
        for level in range(self.n_levels - 1):  # check for each level if tuple is full
            tuple_i = range(first_in_tuple, first_in_tuple + self.n_lights_tuple)
            tuple_complete = np.all(self.state[level, tuple_i])  # check if all lights in tuple_i are on
            next_level_light = first_in_tuple // self.n_lights_tuple
            next_level_off = not self.state[level + 1, next_level_light]  # check if next-level light is off
            if tuple_complete and next_level_off:
                self.state[level, tuple_i] = 0  # turn off lower-level lights
                self.state[level + 1, next_level_light] = 1  # turn on next-level lights
                self.high_lev_change[level, next_level_light] = 1
            first_in_tuple = next_level_light - (next_level_light % self.n_lights_tuple)




            # n_lights_tuple = self.n_lights_tuple ** (1 + level)
            # first_in_tuple = light_i - (light_i % n_lights_tuple)
            # tuple_i = range(first_in_tuple, first_in_tuple + n_lights_tuple)
            # tuple_complete = np.all(self.state[level, tuple])  # check if all lights in a tuple_i are on
            # next_level_off = not np.any(self.state[level + 1, tuple])  # check if next-level lights are still off
            # if tuple_complete and next_level_off:
            #     self.state[level, tuple] = 0  # turn off lower-level lights
            #     self.state[level + 1, tuple] = 1  # turn on next-level lights
            #     self.high_lev_change[level, tuple] = 1
