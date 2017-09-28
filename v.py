import numpy as np


class V(object):
    def __init__(self, env, lambd):
        self.lambd = lambd
        self.initial_value = 1 / env.n_lights_tuple / 2
        self.v = self.initial_value * np.ones([env.n_levels, env.n_lights])  # values of actions and options
        self.v[1:] = np.nan  # undefined for options

    def create_option(self, option):
        self.v[option[0], option[1]] = self.initial_value

    def update(self, agent, option, goal_achieved, learning_signal, events):
        if learning_signal == 'novelty':
            reward_signal = np.exp(-self.lambd * agent.n[option[0], option[1]])
            alpha = 1  # hack to track novelty exactly
        elif learning_signal == 'reward':
            reward_signal = np.sum(events)
            alpha = agent.alpha  # standard RL reward-learning
        else:
            print('Error! Learning signal must either be "novelty" or "reward"!')
        RPE = goal_achieved * reward_signal - self.v[option[0], option[1]]
        self.v[option[0], option[1]] += alpha * RPE

    def get_option_values(self, state, option_stack, theta):
        inside_option = len(option_stack) > 0
        if not inside_option:
            values = self.get()  # select option based on agent.v
        else:  # select option based on in-option policy
            option = option_stack[-1]
            features = 1 - state[option[0] - 1]  # features indicate which lights are OFF
            theta = theta.get_option_thetas(option)
            option_values = np.dot(theta, features)  # calculate values from thetas
            values = np.full(self.get().shape, np.nan)  # initialize value array
            values[option[0] - 1, :] = option_values
        return values

    def get(self):
        return self.v.copy()
