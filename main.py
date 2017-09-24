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
alpha = 0.7  # agent's learning rate
epsilon = 0.2  # inverse of agent's greediness
# gamma = 0.9
distraction = 0.2  # probability option terminates at each step
n_trials = 500  # number of trials in the game
n_levels = math.ceil(n_lights ** (1/n_lights_tuple))  # number of levels (formerly lights of different colors)

# Code for flat agents
lights_on = np.zeros([n_trials, n_agents])
for ag in range(n_agents):
    # print("\n AGENT", ag)
    env = Environment(n_levels, n_lights, n_lights_tuple, n_trials)
    # agent = NoveltyAgentF(alpha, epsilon, env)
    agent = NoveltyAgentH(alpha, epsilon, distraction, env)
    for trial in range(n_trials):
        old_state = env.state.copy()
        action = agent.take_action(old_state)
        env.switch_lights(action)
        events = env.make_events(action)
        new_state = env.state.copy()
        agent.learn(old_state, events)
        agent.trial += 1

# Save value history to csv
trials = np.arange(n_trials)
colnames = ['action' + str(i) for i in range(n_lights)]
v_history = pd.DataFrame(columns=colnames)
for level in range(agent.v.history.shape[0]):
    level_v_history = pd.DataFrame(agent.v.history[level, :, :].transpose(), columns=colnames)
    level_v_history['level'] = level
    level_v_history['trial'] = trials
    v_history = pd.concat([v_history, level_v_history])
v_history.to_csv("C:/Users/maria/MEGAsync/Berkeley/LEARN/data/v_history.csv") # "C:/Users/maria/MEGAsync/Berkeley/LEARN/data"

# Save theta history to csv
colnames = ['action' + str(i) for i in range(n_lights)]
theta_history = pd.DataFrame(columns=colnames)
for option in range(agent.theta.history.shape[0]):
    for feature in range(agent.theta.history.shape[1]):
        option_theta_history = pd.DataFrame(agent.theta.history[option, feature, :, :].transpose(), columns=colnames)
        option_theta_history['option'] = option
        option_theta_history['feature'] = feature
        option_theta_history['trial'] = range(option_theta_history.shape[0])
        theta_history = pd.concat([theta_history, option_theta_history])
theta_history.to_csv("C:/Users/maria/MEGAsync/Berkeley/LEARN/data/theta_history.csv") # "C:/Users/maria/MEGAsync/Berkeley/LEARN/data"



# rownames = ['trial' + str(i) for i in range(n_trials)]
# trials = np.arange(n_trials)
# ax = sns.regplot(x=trials, y="light2", data=v_history, fit_reg=False)
# v_history['trial'] = trials
# v_history_long = pd.melt(v_history, id_vars="trial")
# gr = sns.FacetGrid(v_history_long, hue="variable")
# gr.map(plt.scatter, "trial", "value")
# # gr.map(sns.regplot, "trial", "value")
# gr.add_legend()

# plt.ion()
# plt.figure()
# plt.gca()
# C = plt.Circle((0, 0), .2, color='w')
# C.set_color('r')
# ax = plt.gca()
# ax.add_artist(C)
# plt.xlim(-.5,.5)
# plt.ylim(-.5,.5)
# ax.set_aspect(1)

# lights_on_NF = lights_on
# v_turn_on_NF = v_on
# v_turn_off_NF = v_off
# trials = np.arange(lights_on_NF.shape[0])
# v_on = np.mean(v_turn_on_NF, 1)
# v_off = np.mean(v_turn_off_NF, 1)
#
# plt.figure()
# n_lights_on = np.mean(lights_on_NF, 1)
# ax = sns.regplot(x=trials, y=n_lights_on, fit_reg=False, label="Novelty & reward flat.")
# ax.set(xlabel="Trial", ylabel="Number lights on")
# ax.legend()
#
# plt.figure()
# ax2 = sns.regplot(x=trials, y=v_on, fit_reg=False, label="Value on")
# ax2 = sns.regplot(x=trials, y=v_off, fit_reg=False, label="Value off")
# ax2.set(xlabel="Trial", ylabel="Value")
# ax2.legend()

# # Debugging
# for i in range(n_trials):
#     print(i)
#     old_state = env.state.copy()
#     action = agent.take_action(old_state) #(1, i % n_lights)  #
#     print(action)
#     env.respond(action)
#     if np.sum(env.event) > 0:
#         print("change!")
#         print(agent.o_v)
#         print(agent.o_blocked)
#         print(agent.v)
#         agent.handle_options(env.event, action)
#     agent.update_values(old_state, action, env.state, env.event)
#     # print(action)
#     # print(env.event)
#     # print(env.state)
#     # print(agent.v)
#     # print(agent.level)

