import numpy as np


class FlatAgent(object):
    """
    This class encompasses all flat agents.
    Flat agents do not perceive higher-level lights and cannot create options.
    """
    def __init__(self, alpha, epsilon, n_levels, n_lights, n_lights_tuple):
        self.v = np.zeros([2, n_lights])  # row0: values of turning off; row1: v of turning on; columns: lights
        self.n = np.zeros([2, n_lights]).astype(np.int)
        self.alpha = alpha
        self.epsilon = epsilon

    def take_action(self, state):
        available_values = self.v[1-state[0], range(len(state[0]))]
        best_actions = np.argwhere(available_values == np.max(available_values)).flatten()
        worse_actions = np.argwhere(available_values < np.max(available_values)).flatten()
        if (np.random.rand() > self.epsilon) or (len(worse_actions) == 0):
            selected_a = np.random.choice(best_actions)
        else:
            selected_a = np.random.choice(worse_actions)
        switch_to = 1 - state[0, selected_a]
        return switch_to, selected_a


class RewardAgent(FlatAgent):
    """
    This agent is driven by reward. It perceives reward when lights turn on.
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
    """
    def update_values(self, old_state, action, new_state, high_lev_change):
        self.n[action] += 1
        novelty = 1 / self.n[action]
        reward = sum(new_state[0, :]) - sum(old_state[0, :])
        self.v[action] += self.alpha * (novelty + reward - self.v[action])  # RL with novelty instead reward
