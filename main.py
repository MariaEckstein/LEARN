import math
import numpy as np
from flat_agents import RewardAgent, NoveltyAgentF, NoveltyRewardAgent
from hierarchical_agents import NoveltyAgentH
from environment import Environment

n_lights = 8   # number of level-0 lights (formerly know as "blue" lights); must be n_lights_tuple ** x
n_lights_tuple = 2  # number of lights per level-0 tuple
alpha = 0.75  # agent's learning rate
epsilon = 0.1  # inverse of agent's greediness
n_trials = 22  # number of trials in the game
n_levels = math.ceil(n_lights ** (1/n_lights_tuple))  # number of levels (formerly lights of different colors)

agent = NoveltyAgentH(alpha, epsilon, n_levels, n_lights, n_lights_tuple)
environment = Environment(n_levels, n_lights, n_lights_tuple)

# Debugging
for i in range(n_trials):
    old_state = environment.state.copy()
    action = (1, i % n_lights)  #agent.take_action(old_state) #
    environment.respond(action)
    # [high_lev_change, new_state] = [environment.high_lev_change, environment.state]
    agent.update_values(old_state, action, environment.state, environment.high_lev_change)
    print(action)
    print(environment.high_lev_change)
    print(environment.state)
    print(agent.v)


# The Real Code
# for _ in range(n_trials):
#     old_state = environment.state.copy()
#     print(agent.v)
#     action = agent.take_action(old_state)
#     print(action)
#     environment.respond(action)
#     new_state = environment.state
#     agent.update_values(old_state, action, new_state)
#     print(new_state)