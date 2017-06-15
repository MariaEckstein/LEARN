import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from flat_agents import RewardAgent, NoveltyAgentF, NoveltyRewardAgent
from hierarchical_agents import NoveltyAgentH
from environment import Environment

n_lights = 8   # number of level-0 lights (formerly know as "blue" lights); must be n_lights_tuple ** x
n_lights_tuple = 2  # number of lights per level-0 tuple
n_agents = 1
alpha = 0.75  # agent's learning rate
epsilon = 0.5  # inverse of agent's greediness
distraction = 0.1  # probability option terminates at each step
n_trials = 200  # number of trials in the game
n_levels = math.ceil(n_lights ** (1/n_lights_tuple))  # number of levels (formerly lights of different colors)

# Code for flat agents
lights_on = np.zeros([n_trials, n_agents])
for ag in range(n_agents):
    print(ag)
    # agent = NoveltyAgentF(alpha, epsilon, n_levels, n_lights, n_lights_tuple)
    agent = NoveltyAgentH(alpha, epsilon, distraction, n_levels, n_lights, n_lights_tuple)
    environment = Environment(n_levels, n_lights, n_lights_tuple)
    for trial in range(n_trials):
        old_state = environment.state.copy()
        action = agent.take_action(old_state)
        environment.respond(action)
        new_state = environment.state
        event = environment.event
        # agent.update_values(old_state, action, new_state, event)
        lights_on[trial, ag] = np.sum(new_state[0])
        if np.all(new_state[0]):
            print('Won!')
            break

lights_on_NF = lights_on
v_turn_on_NF = v_on
v_turn_off_NF = v_off
trials = np.arange(lights_on_NF.shape[0])
v_on = np.mean(v_turn_on_NF, 1)
v_off = np.mean(v_turn_off_NF, 1)

plt.figure()
n_lights_on = np.mean(lights_on_NF, 1)
ax = sns.regplot(x=trials, y=n_lights_on, fit_reg=False, label="Novelty & reward flat.")
ax.set(xlabel="Trial", ylabel="Number lights on")
ax.legend()

plt.figure()
ax2 = sns.regplot(x=trials, y=v_on, fit_reg=False, label="Value on")
ax2 = sns.regplot(x=trials, y=v_off, fit_reg=False, label="Value off")
ax2.set(xlabel="Trial", ylabel="Value")
ax2.legend()

# # Debugging
# for i in range(n_trials):
#     print(i)
#     old_state = environment.state.copy()
#     action = agent.take_action(old_state) #(1, i % n_lights)  #
#     print(action)
#     environment.respond(action)
#     if np.sum(environment.event) > 0:
#         print("change!")
#         print(agent.o_v)
#         print(agent.o_blocked)
#         print(agent.v)
#         agent.handle_options(environment.event, action)
#     agent.update_values(old_state, action, environment.state, environment.event)
#     # print(action)
#     # print(environment.event)
#     # print(environment.state)
#     # print(agent.v)
#     # print(agent.level)

