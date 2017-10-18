import numpy as np


class Theta(object):
    def __init__(self, env):
        self.option_coord_to_index = self.__get_coord_function(env)
        self.n_basic_actions = env.n_basic_actions
        self.n_options = np.sum(env.n_options_per_level[1:])

        initial_theta = 1 / env.n_basic_action
        self.theta = np.full([self.n_options, env.n_basic_actions, env.n_basic_actions], np.nan)  # trial x act x feat
        row = 0
        for level in range(1, env.n_levels):
            n_options = env.n_options_per_level[level]
            n_actions = env.n_options_per_level[level-1]
            for option in range(n_options):
                self.theta[row, range(n_actions), :] = initial_theta
                row += 1
        self.e = np.zeros(self.theta.shape)  # Eligibility trace: trial x action x feature
        self.V_old = np.zeros(env.state.shape)

    def get_option_thetas(self, option, action=None):
        if action is None:
            return self.theta[self.option_coord_to_index(option), :, :]  # [option, action, feature]
        else:
            return self.theta[self.option_coord_to_index(option), action, :]

    def update(self, current_option, goal_achieved, agent, hist):
        # Get old thetas, old values, and old phi (features) for this option
        action_level = current_option[0]-1
        old_events = hist.event_s[:, action_level]
        trials = np.argwhere(~np.isnan(old_events))[-2:]
        old_actions = hist.action_s[:, action_level]
        action = int(old_actions[~np.isnan(old_actions)][-1])
        e = self.e[self.option_coord_to_index(current_option)]
        # V_old = self.V_old[current_option[0], current_option[1]]  # scalar
        # Get features, thetas, values for old (before option initiation) and new state (after option termination)
        theta = self.theta[self.option_coord_to_index(current_option)]
        phi = hist.state[trials[0], action_level][0]  # n_levels x n_basic_actions
        phi_prime = hist.state[trials[1], action_level][0]
        V = np.dot(theta[action], phi)
        V_prime = np.dot(theta[action], phi_prime)
        # Update values
        delta = goal_achieved + agent.gamma * V_prime - V
        theta = theta + agent.alpha * delta * e  # should be: theta + agent.alpha (delta + V - V_old) * e - agent.alpha * (V - V_old) * phi
        self.theta[self.option_coord_to_index(current_option)] = theta
        self.V_old[current_option[0], current_option[1]] = V_prime
        # phi = phi_prime

    @staticmethod
    def __get_coord_function(env):
        def option_coord_to_index(coord):
            level, option = coord
            index = np.sum(env.n_options_per_level[1:level]) + option
            return int(index)
        return option_coord_to_index

    def get(self):
        return self.theta.copy()
