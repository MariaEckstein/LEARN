import numpy as np


class Theta(object):
    def __init__(self, env):
        self.option_coord_to_index = self.__get_coord_function(env)
        self.n_basic_actions = env.n_basic_actions
        self.n_options = np.sum(env.n_options_per_level[1:])

        self.initial_theta = 1 / env.n_basic_actions
        self.theta = np.full([self.n_options, env.n_basic_actions, env.n_basic_actions], np.nan)  # option x act x feat
        # self.e = np.full([self.n_options, env.n_basic_actions, env.n_basic_actions], np.nan)  # option x act x feat
        # row = 0
        # for level in range(1, env.n_levels):
        #     n_options = env.n_options_per_level[level]
        #     n_actions = env.n_options_per_level[level-1]
        #     for option in range(n_options):
        #         self.e[row, 0:n_actions, 0:n_actions] = self.initial_theta
        #         row += 1

    def create_option(self, event, env, v):
        option_level = event[0]
        action_level = option_level - 1
        caller_level = option_level + 1
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
                    n_actions = env.n_options_per_level[action_level]
                    self.theta[caller_index, discovered_option, 0:n_actions] = self.initial_theta

    def get_option_thetas(self, option, action=None):
        if action is None:
            return self.theta[self.option_coord_to_index(option), :, :]  # [option, action, feature]
        else:
            return self.theta[self.option_coord_to_index(option), action, :]

    def update(self, agent, hist, current_option, goal_achieved, state_before, state_after):
        action_level = current_option[0] - 1
        actions = hist.event_s[:, action_level]
        action = int(actions[~np.isnan(actions)][-1])
        values_before = agent.v.get_option_values(state_before, current_option, agent.theta)
        v_before = values_before[action_level, action]
        values_after = agent.v.get_option_values(state_after, current_option, agent.theta)
        v_after = max(values_after[action_level, :])  # maxQ
        delta = goal_achieved + agent.gamma * v_after - v_before
        self.theta[self.option_coord_to_index(current_option), action] += agent.alpha * delta
        # e = self.e[self.option_coord_to_index(current_option)]
        # self.theta[self.option_coord_to_index(current_option)] += agent.alpha * delta * e

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

    # def update_e(self, agent, events, hist, env):
    #     current_events = np.argwhere(events)
    #     for event in current_events:
    #         action_level = event[0]
    #         option_level = action_level + 1
    #         if option_level < env.n_levels:
    #             past_actions = np.append(.999, hist.event_s[:, action_level])
    #             past_2_actions = past_actions[~np.isnan(past_actions)][-2:]
    #             phi_before_action = [i == past_2_actions[-2] for i in range(env.n_basic_actions)]
    #             action = int(past_2_actions[-1])
    #             n_options = env.n_options_per_level[option_level]
    #             using_options = [[option_level, i] for i in range(n_options)]
    #             for opt in using_options:
    #                 e_old = self.e[self.option_coord_to_index(opt)]
    #                 phii_old = np.zeros([env.n_basic_actions, env.n_basic_actions])
    #                 phii_old[action] = phi_before_action
    #                 e = agent.gamma * agent.e_lambda * e_old + (1 - agent.alpha * agent.gamma * agent.e_lambda * np.dot(e_old[action], phi_before_action)) * phii_old
    #                 self.e[self.option_coord_to_index(opt)] = e
