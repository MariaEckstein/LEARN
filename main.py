import math
import numpy as np
from flat_agents import RewardAgent, NoveltyAgent, NoveltyRewardAgent
from environment import Environment

n_lights = 32   # number of level-0 lights (formerly know as "blue" lights); must be n_lights_tuple ** x
alpha = 0.75  # agent's learnign rate
epsilon = 0.1  # inverse of agent's greediness
n_trials = 100  # number of trials in the game
n_lights_tuple = 2  # number of lights per level-0 tuple
n_levels = math.ceil(n_lights ** (1/n_lights_tuple))  # number of levels (formerly lights of different colors)


class HierarchicalAgent(object):
    def __init__(self, alpha, epsilon, n_lights, n_levels):
        self.n_options = np.sum([n_lights // n_lights_tuple ** i for i in range(n_levels)])
        self.v = np.zeros([n_levels + 1, n_lights])  # values of actions and options; same organization as state array
        self.n = np.zeros([n_levels + 1, n_lights]).astype(np.int)  # counter for experience with events; same shape
        self.o_v = np.zeros([2, n_lights, self.n_options - n_lights])  # inter-option policies (rows: on/off; cols: lights; depth: options (basic & hier.))
        self.i_option = 0
        self.alpha = alpha
        self.epsilon = epsilon

    def update_values(self, old_state, action, new_state):
        novelty = self.measure_novelty(old_state, action, new_state)
        if novelty >= 1:
            print('This is new!')
        print(novelty)
            # self.create_option(old_state, new_state)
        #
        # update_options()
        # update_action_values()
        # update_option_values()
        # train_options()
        # select_action()

    def measure_novelty(self, old_state, action, new_state):
        # Count how often each basic action has been performed / experienced
        light_i, switch_to = action  # switch_to is either 0 or 1
        self.n[switch_to, light_i] += 1  # only rows 0 and 1 are updated
        novelty_action = 1 / self.n[switch_to, light_i]
        # Count how often each higher-level event has been experienced
        high_lev_change = new_state[1:] - old_state[1:]  # state has one row for each level (DOESN'T WORK YET! SKIPS LEVELS IF THE TURN AND OFF WITHIN ONE TRIAL!! CAN I HAVE DO_EVENT REPORT EVENTS INSTEAD?)
        if np.sum(high_lev_change) > 0:
            self.n[2:] += high_lev_change > 0
            novelty_event = 1 / self.n[2:][high_lev_change == 1][1]
        else:
            novelty_event = 0
        return novelty_action + novelty_event

    # def create_option(self, old_state, new_state):
    #
    #
    #     self.i_option += 1



agent = HierarchicalAgent(alpha, epsilon, n_lights, n_levels)
environment = Environment(n_levels, n_lights, n_lights_tuple)

for i in range(n_trials):
    old_state = environment.state.copy()
    action = [i % n_lights, 1]
    environment.respond(action)
    new_state = environment.state
    agent.update_values(old_state, action, new_state)
    print(agent.n)
    # print(new_state)


# for _ in range(n_trials):
#     old_state = environment.state.copy()
#     print(agent.v)
#     action = agent.take_action(old_state)
#     print(action)
#     environment.respond(action)
#     new_state = environment.state
#     agent.update_values(old_state, action, new_state)
#     print(new_state)