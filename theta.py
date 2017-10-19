import numpy as np


class Theta(object):
    def __init__(self, env):
        self.option_coord_to_index = self.__get_coord_function(env)
        self.n_basic_actions = env.n_basic_actions
        self.n_options = np.sum(env.n_options_per_level[1:])

        initial_theta = 1 / env.n_basic_actions
        self.theta = np.full([self.n_options, env.n_basic_actions, env.n_basic_actions], np.nan)  # option x act x feat
        row = 0
        for level in range(1, env.n_levels):
            n_options = env.n_options_per_level[level]
            n_actions = env.n_options_per_level[level-1]
            for option in range(n_options):
                self.theta[row, 0:n_actions, 0:n_actions] = initial_theta
                row += 1
        self.e = np.round(self.theta.copy())  # Eligibility trace: option x action x feature
        self.V_old = np.zeros(env.state.shape)  # V_old is supposed to be initialized at 0 (p. 262)

    def get_option_thetas(self, option, action=None):
        if action is None:
            return self.theta[self.option_coord_to_index(option), :, :]  # [option, action, feature]
        else:
            return self.theta[self.option_coord_to_index(option), action, :]

    def update(self, current_option, goal_achieved, agent, hist, env):
        option_level = current_option[0]
        action_level = option_level - 1
        past_actions = np.append(.999, hist.event_s[:, action_level])
        past_2_actions = past_actions[~np.isnan(past_actions)][-2:]
        phi_old = np.array([i == past_2_actions[-2] for i in range(env.n_basic_actions)])
        phi_new = np.array([i == past_2_actions[-1] for i in range(env.n_basic_actions)])
        theta = self.theta[self.option_coord_to_index(current_option)]
        e = self.e[self.option_coord_to_index(current_option)]
        V_old = self.V_old[current_option[0], current_option[1]]
        V = np.dot(theta[int(past_2_actions[-1])], phi_old)
        V_new = np.dot(theta[int(past_2_actions[-1])], phi_new)
        delta = goal_achieved + agent.gamma * V_new - V
        # phi_old_full = np.zeros(theta.shape)
        # phi_old_full[np.argwhere(phi_old)] = phi_old  # DOESN'T WORK - NOT SURE WHAT IT'S SUPPOSED TO DO
        theta = theta + agent.alpha * (delta + V_new - V_old) * e   # - agent.alpha * (V_new - V_old) * phi_old_full
        self.theta[self.option_coord_to_index(current_option)] = theta
        self.V_old[current_option[0], current_option[1]] = V_new
        b = 4

    @staticmethod
    def __get_coord_function(env):
        def option_coord_to_index(coord):
            level, option = coord
            index = np.sum(env.n_options_per_level[1:level]) + option
            return int(index)
        return option_coord_to_index

    def get(self):
        return self.theta.copy()
