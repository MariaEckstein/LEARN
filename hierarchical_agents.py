import numpy as np
from theta import Theta
from v import V


# TDs:
# - update eligib. traces for ALL EVENTS!
# - just one set of elig. traces per level of the hierarchy (not every option needs its own elig. tr.)
# - separate lambdas for novelty and eligibility!
# - different learning rates for novelty and values

class Agent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, agent_stuff, env):

        # Agent's RL features
        self.alpha = agent_stuff['alpha'] # learning rate
        self.epsilon = agent_stuff['epsilon']  # greediness
        self.e_lambda = agent_stuff['e_lambda']
        self.n_lambda = agent_stuff['n_lambda']
        self.gamma = agent_stuff['gamma']
        self.distraction = agent_stuff['distraction']  # propensity to quit unfinished options
        self.hier_level = agent_stuff['hier_level']  # what is the highest-level option the agent can select?
        self.learning_signal = agent_stuff['learning_signal']

        # Agent's thoughts about its environment (novelty, values, theta)
        self.n = np.zeros([env.n_levels, env.n_basic_actions])  # event counter = inverse novelty
        self.trial = 0  # current trial
        self.v = V(env, self.n_lambda)
        self.theta = Theta(env)

        # Agent's plans and memory about his past actions
        self.option_stack = []  # stack of the option(s) that are currently guiding behavior
        self.option_history = np.zeros([env.n_trials * env.n_levels, env.n_levels, env.n_basic_actions + 2])
        self.option_row = 0

    def take_action(self, old_state, hist, env):
        hist.v[self.trial, :, :] = self.v.get()
        values = self.v.get_option_values(old_state, self.option_stack, self.theta)
        option = self.__select_option(values)
        self.option_stack.append(option)
        hist.action_s[self.trial, option[0]] = option[1]
        if self.__is_basic(option):
            return option  # execute option
        else:
            return self.take_action(old_state, hist, env)  # use option policy to select next option(s)

    def __select_option(self, values):
        option = [1000, 1000]
        while option[0] > self.hier_level:  # keep drawing until option comes from an allowed level
            if self.__is_greedy():
                selected_options = np.argwhere(values == np.nanmax(values))  # all options with the highest value
            else:
                selected_options = np.argwhere(~np.isnan(values))  # all options that are not nan
            select = np.random.randint(len(selected_options))  # randomly select the index of one of the options
            option = selected_options[select]  # pick that option
        return option

    def __is_greedy(self):
        return np.random.rand() > self.epsilon

    @staticmethod
    def __is_basic(option):
        return option[0] == 0

    def __is_novel(self, event):
        return self.n[event[0], event[1]] == 1

    def learn(self, events, hist, env):
        if self.hier_level > 0:
            self.__update_option_history()
        self.__update_elig_traces(events, hist, env)
        self.__learn_from_events(events, hist, env)  # count events, initialize new options (v & theta)
        self.__learn_rest(events, hist, env)  # update theta of ongoing options, v of terminated

    def __update_elig_traces(self, events, hist, env):
        current_events = np.argwhere(events)
        for event in current_events:
            action_level = event[0]
            option_level = action_level + 1
            if option_level < env.n_levels:
                past_actions = np.append(.999, hist.event_s[:, action_level])
                past_2_actions = past_actions[~np.isnan(past_actions)][-2:]
                phi = [i == past_2_actions[-2] for i in range(env.n_basic_actions)]
                action = int(past_2_actions[-1])
                n_options = env.n_options_per_level[option_level]
                using_options = [[option_level, i] for i in range(n_options)]
                for opt in using_options:
                    e_old = self.theta.e[self.theta.option_coord_to_index(opt)]
                    e_new = np.zeros([env.n_basic_actions, env.n_basic_actions])
                    e_new[action] = phi
                    e = self.gamma * self.e_lambda * e_old + (1 - self.alpha * self.gamma * self.e_lambda) * e_new
                    self.theta.e[self.theta.option_coord_to_index(opt)] = e
                    a = 4

    def __learn_from_events(self, events, hist, env):
        current_events = np.argwhere(events)
        for event in current_events:
            self.n[event[0], event[1]] += 1  # update event count (novelty decreases)
            if self.__is_novel(event):
                self.v.create_option(event)
                self.v.update(self, event, 1, self.learning_signal, events)  # update option value right away
            if not self.__is_basic(event):  # it's a higher-level event
                hist.update_theta_history(self, event)
                self.theta.update(event, 1, self, hist, env)  # update in-option policy
    #
    # def __update_theta_history(self, trial, option):
    #     self.theta.history[self.theta.h_row, :, :, :self.theta.n_lights] = self.theta.get()
    #     self.theta.history[self.theta.h_row, :, :, -2] = trial
    #     self.theta.history[self.theta.h_row, :, :, -1] = self.theta.option_coord_to_index(option)
    #     self.theta.h_row += 1

    def __update_option_history(self):
        self.step = 0
        for current_option in self.option_stack:  # list all current options
            self.option_history[self.option_row, current_option[0], current_option[1]] = 1
            self.option_history[self.option_row, :, -2] = self.trial
            self.option_history[self.option_row, :, -1] = self.step
            self.step += 1
            self.option_row += 1

    def __learn_rest(self, events, hist, env):
        current_events = np.argwhere(events)
        for current_option in np.flipud(self.option_stack):  # go through options, starting at lowest-level one
            # Rest
            [goal_achieved, distracted] = self.__get_goal_achieved_distracted(current_option, current_events)
            if not self.__is_basic(current_option) and not goal_achieved:  # thetas not covered by learn_from_events
                hist.update_theta_history(self, current_option)
                self.theta.update(current_option, 0, self, hist, env)
            if goal_achieved:  # goal achieved -> event happened -> update expected novelty toward perceived novelty
                if not self.__is_novel(current_option):  # novel events are already updated in learn_from_events
                    self.v.update(self, current_option, 1, self.learning_signal, events)
                previous_option = self.option_stack.pop()
            elif distracted:  # goal not achieved -> event didn't happen -> update expected novelty toward 0
                self.v.update(self, current_option, 0, self.learning_signal, events)
                previous_option = self.option_stack.pop()
            else:  # if current_option has not terminated, no higher-level option can have terminated
                break  # no more updating needed

    def __get_goal_achieved_distracted(self, option, events):
        goal_achieved = any([np.all(option == event) for event in events])  # did option's target event occur?
        distracted = self.distraction > np.random.rand()
        return [goal_achieved, distracted]
