import numpy as np


class Theta(object):
    def __init__(self, env):
        self.option_coord_to_index = self.__get_coord_function(env)
        self.n_options = np.sum([env.n_lights // env.n_lights_tuple ** i for i in range(env.n_levels)])
        self.n_lights = env.n_lights

        initial_theta = 1 / env.n_lights / 2
        self.theta = np.full([self.n_options - env.n_lights + 1, env.n_lights, env.n_lights], np.nan)  # in-option features
        row_ot = 0
        for level in range(env.n_levels):
            n_options_level = env.n_lights // (env.n_lights_tuple * env.n_lights_tuple ** level)
            for option in range(n_options_level):
                n_lights_level = env.n_lights // env.n_lights_tuple ** level  # number of lights at level below option
                self.theta[row_ot, range(n_lights_level), :] = initial_theta
                row_ot += 1
        theta_shape = list(self.theta.shape)
        theta_shape[-1] += 1
        theta_shape.insert(0, env.n_levels * env.n_trials)
        self.history = np.zeros(theta_shape)
        self.h_row = 0

    def get_option_thetas(self, option, index=None):
        if index is None:
            return self.theta[self.option_coord_to_index(option), :, :]
        else:
            return self.theta[self.option_coord_to_index(option), index, :]

    def update(self, current_option, goal_achieved, old_state, previous_option, agent):
        theta_previous_option = self.get_option_thetas(current_option, previous_option[1])
        features = 1 - old_state[current_option[0]-1]
        value_previous_option = np.dot(theta_previous_option, features)
        RPE = goal_achieved - value_previous_option
        theta_previous_option += agent.alpha * RPE / sum(features) * features

    @staticmethod
    def __get_coord_function(env):
        def option_coord_to_index(coord):
            level, selected_a = coord
            if level == 0:
                n_options_below = - env.n_lights
            else:
                n_tuples_level = env.n_lights // env.n_lights_tuple ** level
                n_options_below = np.sum([n_tuples_level * env.n_lights_tuple ** i for i in range(1, level)])
            index = n_options_below + selected_a
            return int(index)
        return option_coord_to_index

    def get(self):
        return self.theta.copy()
