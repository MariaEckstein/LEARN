import numpy as np


class Environment(object):
    def __init__(self, n_levels, n_lights, n_lights_tuple, n_trials):
        self.n_trials = n_trials
        self.n_levels = n_levels
        self.n_lights = n_lights
        self.n_lights_tuple = n_lights_tuple
        self.state = np.zeros((self.n_levels, self.n_lights), dtype=bool)
        self.events = np.zeros((self.n_levels, self.n_lights), dtype=bool)

    def switch_lights(self, action):
        self.state[action[0], action[1]] = 1
        return self.state.copy()

    def make_events(self, action):
        self.events[:] = 0
        # record basic events
        self.events[action[0], action[1]] = 1
        # check for events on higher levels
        first_in_tuple = action[1] - (action[1] % self.n_lights_tuple)
        for level in range(self.n_levels - 1):  # check for each level if tuple is full
            tuple_i = range(first_in_tuple, first_in_tuple + self.n_lights_tuple)
            tuple_complete = np.all(self.state[level, tuple_i])  # check if all lights in tuple_i are on
            next_level_light = first_in_tuple // self.n_lights_tuple
            # next_level_off = not self.state[level + 1, next_level_light]  # check if next-level light is off
            if tuple_complete:  # and next_level_off:
                self.state[level, tuple_i] = 0  # turn off lower-level lights
                self.state[level + 1, next_level_light] = 1  # turn on next-level light
                self.events[level + 1, next_level_light] = 1  # note this event
            first_in_tuple = next_level_light - (next_level_light % self.n_lights_tuple)
        return self.events.copy()
