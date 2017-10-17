import numpy as np


class Theta(object):
    def __init__(self, env):
        self.option_coord_to_index = self.__get_coord_function(env)
        self.n_options = np.sum([env.n_basic_actions // env.option_length ** i for i in range(env.n_levels)])
        self.n_lights = env.n_basic_actions

        initial_theta = 1 / env.n_basic_actions / 2
        self.theta = np.full([self.n_options - env.n_basic_actions + 1, env.n_basic_actions, env.n_basic_actions], np.nan)  # in-option features
        row_ot = 0
        for level in range(env.n_levels):
            n_options_level = env.n_basic_actions // (env.option_length * env.option_length ** level)
            for option in range(n_options_level-1):
                n_lights_level = env.n_basic_actions // env.option_length ** level  # number of lights at level below option
                self.theta[row_ot, range(n_lights_level), :] = initial_theta
                row_ot += 1

    def get_option_thetas(self, option, action=None):
        if action is None:
            return self.theta[self.option_coord_to_index(option), :, :]  # [option, action, feature]
        else:
            return self.theta[self.option_coord_to_index(option), action, :]

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
                n_options_below = - env.n_basic_actions
            else:
                n_tuples_level = env.n_basic_actions // env.option_length ** level
                n_options_below = np.sum([n_tuples_level * env.option_length ** i for i in range(1, level)])
            index = n_options_below + selected_a
            return int(index)
        return option_coord_to_index

    def get(self):
        return self.theta.copy()
