import numpy as np


class Environment(object):
    def __init__(self, n_levels, n_lights, n_lights_tuple):
        self.n_levels = n_levels
        self.n_blue_lights = n_lights
        self.n_lights_tuple = n_lights_tuple
        self.state = np.zeros((n_levels, n_lights)).astype(np.int)

    def respond(self, action):
        light_i, switch_to = action  # action in tuple-form: action = (light_i, switch_to)
        self.state[0, light_i] = switch_to
        self.do_events(light_i)

    def do_events(self, light_i):
        for level in range(self.n_levels - 1):  # check for each level if rules fulfilled (blue, yellow, green, etc.)
            n_lights_tuple = self.n_lights_tuple * 3 ** level
            first_in_tuple = light_i - (light_i % n_lights_tuple)
            color_tuple = range(first_in_tuple, first_in_tuple + n_lights_tuple)
            if all(self.state[level, color_tuple]):  # if all lights in a tuple of this level are on
                self.state[level + 1, color_tuple] = True  # turn next-level light on; take multiple columns
                self.state[level, color_tuple] = 0  # turn lower-level lights off
