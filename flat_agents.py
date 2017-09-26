import numpy as np


class FlatAgent(object):
    """
    This class encompasses all flat agents.
    Flat agents do not perceive higher-level lights and cannot create options.
    """
    def __init__(self, alpha, epsilon, n_levels, n_lights, n_lights_tuple):
        self.v = np.random.rand(n_lights)  # values of each basic action
        self.n = np.zeros(n_lights).astype(np.int)
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_lights = n_lights

    def __select_option(self, values):
        if self.__is_greedy():
            selected_options = np.argwhere(values == np.nanmax(values))  # all options with the highest value
        else:
            selected_options = np.argwhere(~ np.isnan(values))  # all options that are not nan
        select = np.random.randint(len(selected_options))  # randomly select the index of one of the options
        option = selected_options[select]  # pick that option
        return option

    def __is_greedy(self):
        return np.random.rand() > self.epsilon

    def take_action(self, state):
        if np.random.rand() > self.epsilon:
            action = np.argmax(self.v)
        else:
            action = np.random.choice(range(self.n_lights))
        return action


class RewardAgent(FlatAgent):
    """
    This agent is driven by reward. It perceives reward when lights turn on.
    MAKES NO SENSE RIGHT NOW BECAUSE AGENTS ARE NOT SWITCHING ANY LIGHTS OFF.
    SO THE STATE OF A BASIC ACTION CAN BE TURNED ON MULTIPLE TIMES.
    """
    def update_values(self, old_state, action, new_state, high_lev_change):
        reward = sum(new_state[0, :]) - sum(old_state[0, :])
        self.v[action] += self.alpha * (reward - self.v[action])  # classic RL value update


class NoveltyAgentF(FlatAgent):
    """
    This agent is driven by novelty. The novelty of his actions decreases exponentially after the first time executed.
    """
    def update_values(self, old_state, action, new_state, high_lev_change):
        self.n[action] += 1
        novelty = 1 / self.n[action]
        self.v[action] += self.alpha * (novelty - self.v[action])  # RL with novelty instead reward


class NoveltyRewardAgent(FlatAgent):
    """
    This agent is driven by novelty and reward. It's a combination of the RewardAgent and the NoveltyAgent.
    MAKES NO SENSE RIGHT NOW BECAUSE AGENTS ARE NOT SWITCHING ANY LIGHTS OFF.
    SO THE STATE OF A BASIC ACTION CAN BE TURNED ON MULTIPLE TIMES.
    """
    def update_values(self, old_state, action, new_state, high_lev_change):
        self.n[action] += 1
        novelty = 1 / self.n[action]
        reward = sum(new_state[0, :]) - sum(old_state[0, :])
        self.v[action] += self.alpha * (novelty + reward - self.v[action])  # RL with novelty instead reward
