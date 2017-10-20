import numpy as np


class Theta(object):
    def __init__(self, env):
        self.option_coord_to_index = self.__get_coord_function(env)
        self.n_basic_actions = env.n_basic_actions
        self.n_options = np.sum(env.n_options_per_level[1:])

        self.initial_theta = 1 / env.n_basic_actions
        self.theta = np.full([self.n_options, env.n_basic_actions, env.n_basic_actions], np.nan)  # option x act x feat
        self.e = np.full([self.n_options, env.n_basic_actions, env.n_basic_actions], np.nan)  # option x act x feat
        row = 0
        for level in range(1, env.n_levels):
            n_options = env.n_options_per_level[level]
            n_actions = env.n_options_per_level[level-1]
            for option in range(n_options):
                self.e[row, 0:n_actions, 0:n_actions] = self.initial_theta
                row += 1
        self.V_old = np.zeros(env.state.shape)  # V_old is supposed to be initialized at 0 (p. 262)

    def create_option(self, event, env, v):
        option_level = event[0]
        action_level = option_level - 1
        # Fill up theta table of newly-encountered option
        if action_level >= 0:  # only for options (i.e., action level exists)
            n_actions = env.n_options_per_level[action_level]
            discovered_actions = np.argwhere(~np.isnan(v[action_level]))
            option_index = self.option_coord_to_index(event)
            self.theta[option_index, discovered_actions, 0:n_actions] = self.initial_theta
        # Add newly-encountered action to all options that could use it
        if action_level > 0:  # only for options based on options (i.e., action level > 0)
            discovered_options = np.argwhere(~np.isnan(v[option_level]))
            for option in discovered_options:
                discovered_action = event
                option_index = self.option_coord_to_index([option_level, option])
                n_actions = env.n_options_per_level[action_level]
                self.theta[option_index, discovered_action, 0:n_actions] = self.initial_theta

    def get_option_thetas(self, option, action=None):
        if action is None:
            return self.theta[self.option_coord_to_index(option), :, :]  # [option, action, feature]
        else:
            return self.theta[self.option_coord_to_index(option), action, :]

    def update_e(self, agent, events, hist, env):
        current_events = np.argwhere(events)
        for event in current_events:
            action_level = event[0]
            option_level = action_level + 1
            if option_level < env.n_levels:
                past_actions = np.append(.999, hist.event_s[:, action_level])
                past_2_actions = past_actions[~np.isnan(past_actions)][-2:]
                phi_old = [i == past_2_actions[-2] for i in range(env.n_basic_actions)]
                action = int(past_2_actions[-1])
                n_options = env.n_options_per_level[option_level]
                using_options = [[option_level, i] for i in range(n_options)]
                for opt in using_options:
                    e_old = self.e[self.option_coord_to_index(opt)]
                    phii_old = np.zeros([env.n_basic_actions, env.n_basic_actions])
                    phii_old[action] = phi_old
                    e = agent.gamma * agent.e_lambda * e_old + (1 - agent.alpha * agent.gamma * agent.e_lambda * np.dot(e_old[action], phi_old)) * phii_old
                    self.e[self.option_coord_to_index(opt)] = e

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

    @staticmethod
    def __get_coord_function(env):
        def option_coord_to_index(coord):
            level, option = coord
            if level == 0:
                index = np.nan
            else:
                index = int(np.sum(env.n_options_per_level[1:level]) + option)
            return index
        return option_coord_to_index

    def get(self):
        return self.theta.copy()
