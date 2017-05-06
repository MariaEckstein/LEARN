from collections import defaultdict
import numpy as np
import math
n_blue_lights = 9
n_lights_lev0_tuple = 3
n_trials = 200

n_levels = math.ceil(n_blue_lights ** (1/n_lights_lev0_tuple))
alpha = 0.75
epsilon = 0.1


class Environment(object):
    def __init__(self):
        self.state = np.zeros((n_levels, n_blue_lights)).astype(np.int)

    def respond(self, action):
        light_i, switch_to = action  # action in tuple-form: action = (light_i, switch_to)
        self.state[0, light_i] = switch_to
        self.do_events(light_i)

    def do_events(self, light_i):
        n_levels = self.state.shape[0] - 1
        for level in range(n_levels):  # check for each level if rules fulfilled (blue, yellow, green, etc.)
            n_lights_tuple = n_lights_lev0_tuple * 3 ** level
            first_in_tuple = light_i - (light_i % n_lights_tuple)
            color_tuple = range(first_in_tuple, first_in_tuple + n_lights_tuple)
            if all(self.state[level, color_tuple]):  # if all lights in a tuple of this level are on
                self.state[level + 1, color_tuple] = True  # turn next-level light on; take multiple columns
                self.state[level, color_tuple] = 0  # turn lower-level lights off


class FlatAgent(object):
    def __init__(self):
        self.v = np.zeros([2, n_blue_lights])  # row0: values of turning off; row1: v of turning on; columns: lights
        self.n = np.zeros([2, n_blue_lights]).astype(np.int)

    def take_action(self, state):
        available_values = self.v[1-state[0], range(n_blue_lights)]
        best_actions = np.argwhere(available_values == np.max(available_values)).flatten()
        worse_actions = np.argwhere(available_values < np.max(available_values)).flatten()
        if (np.random.rand() > epsilon) | (len(worse_actions) == 0):
            light_i = np.random.choice(best_actions)
        else:
            light_i = np.random.choice(worse_actions)
        switch_to = 1 - state[0, light_i]
        return light_i, switch_to


class RewardAgent(FlatAgent):
    def update_values(self, old_state, action, new_state):
        light_i, switch_to = action
        reward = sum(new_state[0, :]) - sum(old_state[0, :])
        self.v[switch_to, light_i] += alpha * (reward - self.v[switch_to, light_i])  # classic RL value update


class NoveltyAgent(FlatAgent):
    def update_values(self, old_state, action, new_state):
        light_i, switch_to = action
        self.n[switch_to, light_i] += 1
        novelty = 1 / self.n[switch_to, light_i]
        self.v[switch_to, light_i] += alpha * (novelty - self.v[switch_to, light_i])  # RL with novelty instead reward


class NoveltyRewardAgent(FlatAgent):
    def update_values(self, old_state, action, new_state):
        light_i, switch_to = action
        self.n[switch_to, light_i] += 1
        novelty = 1 / self.n[switch_to, light_i]
        reward = sum(new_state[0, :]) - sum(old_state[0, :])
        self.v[switch_to, light_i] += alpha * (novelty + reward - self.v[switch_to, light_i])  # RL with novelty instead reward


agent = NoveltyRewardAgent()
environment = Environment()

for _ in range(n_trials):
    old_state = environment.state.copy()  # ohne copy waeren old_state und new_state nur andere namen fuer environment.state
    print(agent.v)
    action = agent.take_action(old_state)
    print(action)
    environment.respond(action)
    new_state = environment.state
    agent.update_values(old_state, action, new_state)
    print(new_state)