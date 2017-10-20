import numpy as np


class V(object):
    def __init__(self, env, n_lambda):
        self.n_lambda = n_lambda
        self.initial_value = 1
        self.v = self.initial_value * np.ones([env.n_levels, env.n_basic_actions])  # level x action
        self.v[1:] = np.nan  # undefined for options
        self.step_counter = np.zeros(self.v.shape)

    def create_option(self, option):
        self.v[option[0], option[1]] = self.initial_value

    def update(self, agent, option, goal_achieved, events):
        if agent.learning_signal == 'novelty':
            event_novelty = np.exp(-self.n_lambda * agent.n[option[0], option[1]])
            steps_till_event_reached = self.step_counter[option[0], option[1]]
            reward_signal = agent.gamma ** steps_till_event_reached * event_novelty
        elif agent.learning_signal == 'reward':
            reward_signal = np.sum(events)
        RPE = goal_achieved * reward_signal - self.v[option[0], option[1]]
        self.v[option[0], option[1]] += agent.alpha * RPE

    def get_option_values(self, state, option_stack, theta):
        inside_option = len(option_stack) > 0
        if not inside_option:
            values = self.get()  # select option based on agent.v
        else:  # select option based on in-option policy
            option = option_stack[-1]
            action_level = option[0] - 1
            features = state[action_level]
            theta = theta.get_option_thetas(option)
            action_values = np.dot(theta, features)  # calculate values from thetas
            values = np.full(self.get().shape, np.nan)  # initialize value array
            values[action_level, :] = action_values
        return values

    def get(self):
        return self.v.copy()
