import math
import numpy as np
from flat_agents import RewardAgent, NoveltyAgent, NoveltyRewardAgent
from environment import Environment

n_lights = 18   # number of level-0 lights
alpha = 0.75
epsilon = 0.1
n_trials = 200
n_lights_tuple = 3  # number of lights per level-0 tuple
n_levels = math.ceil(n_lights ** (1/n_lights_tuple))


# class HierarchicalAgent(object):
#     def __init__(self, alpha, epsilon, n_lights):
#         self.v = np.zeros([2, n_lights])  # row0: values of turning off; row1: v of turning on; columns: lights
#         self.n = np.zeros([2, n_lights]).astype(np.int)
#         self.alpha = alpha
#         self.epsilon = epsilon
#
#     def take_action(self, state):
#         if event:
#             create_option(event)
#
#         update_options()
#         update_action_values()
#         update_option_values()
#         train_options()
#         select_action()
#
#     def create_option(self, event):



agent = NoveltyRewardAgent(alpha, epsilon, n_lights)
environment = Environment(n_levels, n_lights, n_lights_tuple)

for _ in range(n_trials):
    old_state = environment.state.copy()
    print(agent.v)
    action = agent.take_action(old_state)
    print(action)
    environment.respond(action)
    new_state = environment.state
    agent.update_values(old_state, action, new_state)
    print(new_state)