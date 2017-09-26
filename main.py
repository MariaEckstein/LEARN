import math
import numpy as np
import pandas as pd
from ggplot import *
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")
from flat_agents import RewardAgent, NoveltyAgentF, NoveltyRewardAgent
from hierarchical_agents import NoveltyAgentH
from environment import Environment

n_lights = 27   # number of level-0 lights (formerly know as "blue" lights); must be n_lights_tuple ** x
n_lights_tuple = 3  # number of lights per level-0 tuple
n_agents = 1
alpha = 0.7  # agent's learning rate
epsilon = 0.2  # inverse of agent's greediness
distraction = 0  # probability option terminates at each step
n_trials = 50  # number of trials in the game
n_levels = int(math.ceil(n_lights ** (1/n_lights_tuple)))  # number of levels (formerly lights of different colors)

for ag in range(n_agents):
    # print("\n AGENT", ag)
    env = Environment(n_levels, n_lights, n_lights_tuple, n_trials)
    # agent = NoveltyAgentF(alpha, epsilon, env)
    agent = NoveltyAgentH(alpha, epsilon, distraction, env)
    for trial in range(n_trials):
        old_state = env.state.copy()
        env.state_history[trial, :, :] = old_state
        action = agent.take_action(old_state)
        agent.action_history[trial, action[1]] = 1
        env.switch_lights(action)
        events = env.make_events(action)
        env.event_history[trial, :, :] = events
        new_state = env.state.copy()
        agent.learn(old_state, events)
        agent.trial += 1

# Save event history to csv
colnames = [str(i) for i in range(n_lights)]
event_history = pd.DataFrame(columns=colnames)
for trial in range(env.event_history.shape[0]):
    level_event_history = pd.DataFrame(env.event_history[trial, :, :], columns=colnames)
    level_event_history['trial'] = trial
    level_event_history['level'] = range(n_levels)
    event_history = pd.concat([event_history, level_event_history])
event_history_long = pd.melt(event_history, id_vars=["trial", "level"], var_name="action")
event_history_long['action'] = pd.to_numeric(event_history_long['action'])
event_history_long.to_csv("C:/Users/maria/MEGAsync/Berkeley/LEARN/data/event_history_long.csv")

# Save value history to csv
trials = np.arange(n_trials)
colnames = [str(i) for i in range(n_lights)]
v_history = pd.DataFrame(columns=colnames)
for trial in range(agent.v.history.shape[0]):
    level_v_history = pd.DataFrame(agent.v.history[trial, :, :], columns=colnames)
    level_v_history['trial'] = trial
    level_v_history['level'] = range(n_levels)
    v_history = pd.concat([v_history, level_v_history])
v_history.head()
v_history_long = pd.melt(v_history, id_vars=["level", "trial"], var_name="action")
v_history_long['action'] = pd.to_numeric(v_history_long['action'])
ident = ""  # "_n_lights" + str(n_lights) + "_n_lights_tuple" + str(n_lights_tuple) + "_alpha" + str(alpha)
v_history_long.to_csv("C:/Users/maria/MEGAsync/Berkeley/LEARN/data/v_history_long" + ident + ".csv")

# Save option history to csv
colnames = [str(i) for i in range(n_lights)]
colnames = colnames + ['trial', 'step']
option_history = pd.DataFrame(columns=colnames)
for row in range(agent.option_history.shape[0]):
    step_history = pd.DataFrame(agent.option_history[row, :, :], columns=colnames)
    step_history['level'] = range(n_levels)
    option_history = pd.concat([option_history, step_history])
option_history.head()
option_history_long = pd.melt(option_history, id_vars=["trial", "step", "level"], var_name="action")
option_history_long['action'] = pd.to_numeric(option_history_long['action'])
option_history_long.to_csv("C:/Users/maria/MEGAsync/Berkeley/LEARN/data/option_history_long.csv")

# Save theta history to csv
colnames = [str(i) for i in range(n_lights)]
colnames = colnames + ['trial']
theta_history = pd.DataFrame(columns=colnames)
for row in range(agent.theta.history.shape[0]):
    for option in range(agent.theta.history.shape[1]):
        option_theta_history = pd.DataFrame(agent.theta.history[row, option, :, :], columns=colnames)
        option_theta_history['option'] = option
        option_theta_history['feature'] = range(n_lights)
        theta_history = pd.concat([theta_history, option_theta_history])
theta_history.head()
theta_history = theta_history[theta_history['1'] != 0]
theta_history_long = pd.melt(theta_history, id_vars=["trial", "option", "feature"], var_name="action")
theta_history_long['action'] = pd.to_numeric(theta_history_long['action'])
theta_history_long.to_csv("C:/Users/maria/MEGAsync/Berkeley/LEARN/data/theta_history_long.csv")


# rownames = ['trial' + str(i) for i in range(n_trials)]
# trials = np.arange(n_trials)
# ax = sns.regplot(x=trials, y="light2", data=v_history, fit_reg=False)
# v_history['trial'] = trials
# v_history_long = pd.melt(v_history, id_vars="trial")
# gr = sns.FacetGrid(v_history_long, hue="variable")
# gr.map(plt.scatter, "trial", "value")
# # gr.map(sns.regplot, "trial", "value")
# gr.add_legend()

# #.loc[event_history_long['trial'] == 0]
# ggplot(event_history_long, aes('variable', 'level')) +\
#     geom_tile(aes(fill='value'))  # , color='white'
#     # geom_point() +\

# ggplot(event_history_long, aes('trial', 'value')) +\
#     geom_point()
#     geom_tile(aes(fill='value'), color='white')

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

