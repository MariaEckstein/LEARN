import numpy as np


class V(object):
    def __init__(self, env, n_lambda):
        self.n_lambda = n_lambda
        self.initial_value = 1 / env.n_basic_actions
        self.c = self.initial_value * np.ones([env.n_levels, env.n_basic_actions])  # level x action
        self.c[1:] = np.nan  # undefined for options
        self.r = np.zeros([env.n_levels, env.n_basic_actions])  # level x action
        self.r[1:] = np.nan
        self.step_counter = np.zeros(self.c.shape)

    def create_option(self, option):
        self.c[option[0], option[1]] = self.initial_value
        self.r[option[0], option[1]] = 0

    def learn_from_rewards(self, rewards, alpha):
        delta = (rewards - self.r) * (rewards > 0)  # discrepancy at rewarded locations only
        self.r += alpha * delta

    def update_c(self, agent, option, goal_achieved, events):
        if agent.learning_signal == 'novelty':
            if agent.hier_level > 0:
                steps_till_event_reached = self.step_counter[option[0], option[1]]
                event_novelty = np.exp(-self.n_lambda * agent.n[option[0], option[1]])
            else:
                steps_till_event_reached = 0
                event_novelties = np.exp(-self.n_lambda * np.dot(np.transpose(events), agent.n))  # cur current events
                event_novelty = np.sum(event_novelties[event_novelties < 1])  # only events that have already happened
            reward_signal = agent.gamma ** steps_till_event_reached * event_novelty
        elif agent.learning_signal == 'reward':
            reward_signal = np.sum(events)
        delta = goal_achieved * reward_signal - self.c[option[0], option[1]]
        self.c[option[0], option[1]] += agent.alpha * delta

    def get_option_values(self, state, option, theta):
        action_level = option[0] - 1
        phi = state[action_level]
        thetas = theta.get_option_thetas(option)
        action_values = np.nansum(thetas * phi, 1)  # same as np.dot(thetas, phi) but ignores nan
        action_values[action_values == 0] = np.nan  # np.nansum replaces actual nans by 0
        values = np.full(self.get().shape, np.nan)
        values[action_level, :] = action_values
        return values.copy()

    def get(self):
        return self.c + self.r
