import numpy as np


class Theta(object):
    def __init__(self, env):
        self.option_coord_to_index = self.__get_coord_function(env)
        self.n_basic_actions = env.n_basic_actions
        self.n_options = np.sum(env.n_options_per_level[1:])
        self.initial_theta = 1 / env.n_basic_actions
        self.theta = np.full([self.n_options, env.n_basic_actions, env.n_basic_actions], np.nan)  # option x act x feat

    def create_option(self, event, env, v):
        action_level = event[0] - 1
        option_level = event[0]
        caller_level = event[0] + 1
        # Fill up theta table of newly-encountered option
        if action_level >= 0:  # only for options (i.e., action level exists)
            n_actions = env.n_options_per_level[action_level]
            discovered_actions = np.argwhere(~np.isnan(v[action_level]))
            option_index = self.option_coord_to_index(event)
            self.theta[option_index, discovered_actions, 0:n_actions] = self.initial_theta
        # Add newly-encountered option to all caller options that could use it
        if option_level > 1 and caller_level < env.n_levels:  # only for options based on options
            caller_options = np.argwhere(~np.isnan(v[caller_level]))
            if len(caller_options) > 0:
                for caller in caller_options:
                    discovered_option = event
                    caller_index = self.option_coord_to_index([caller_level, caller])
                    n_options = env.n_options_per_level[option_level]
                    self.theta[caller_index, discovered_option, 0:n_options] = self.initial_theta

    def update(self, agent, hist, current_option, goal_achieved, state_before, state_after):
        action_level = current_option[0] - 1
        actions = hist.event_s[:, action_level]
        action = int(actions[~np.isnan(actions)][-1])
        values_before = agent.v.get_option_values(state_before, current_option, agent.theta)
        v_before = values_before[action_level, action]
        if not goal_achieved:
            values_after = agent.v.get_option_values(state_after, current_option, agent.theta)
            v_after = max(values_after[action_level, :])  # maxQ
        else:
            v_after = 0
        delta = goal_achieved + agent.gamma * v_after - v_before
        self.theta[self.option_coord_to_index(current_option), action, state_before[action_level]] += agent.alpha * delta

    def get_option_thetas(self, option, action=None):
        if action is None:
            return self.theta[self.option_coord_to_index(option), :, :].copy()  # [option, action, feature]
        else:
            return self.theta[self.option_coord_to_index(option), action, :].copy()

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
