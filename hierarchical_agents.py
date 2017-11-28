import numpy as np
from theta import Theta
from v import V


# TDs:
# - read RL with Hierarchies of machines; MaxQ; The Option-Critic Architecture
# - check if other people have had more than 2 levels of options
# - leave a light on at each level
# - compare to stronger baseline: rewards depending on number of lights turned on also take into accoutn the level
# - check if i'm counting the number of steps right (steps_till_event_reached) -> it's weird that level-0 actions have lower curiosity than higher-level actions
# - different learning rates for novelty and values  => maybe later

class Agent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, parameters, learning_signal, hier_level, id, env):
        self.learning_signal = learning_signal
        self.hier_level = hier_level
        self.id = id
        # Agent's RL features
        self.alpha = parameters['alpha']  # learning rate
        self.epsilon = parameters['epsilon']  # greediness
        self.n_lambda = parameters['n_lambda']  # decay rate of novelty
        self.gamma = parameters['gamma']  # 1 - future discounting
        self.distraction = parameters['distraction']  # propensity to quit unfinished options
        # Agent's thoughts about its environment (novelty, values, theta)
        self.n = np.zeros([env.n_levels, env.n_basic_actions])  # event counter
        self.v = V(env, self.n_lambda)  # curiosity about basic actions and already-discovered options
        self.theta = Theta(env)  # feature weights
        # Agent's plans and memory about his past actions
        self.option_stack = []  # stack of the option(s) that are currently guiding behavior

    # Take_action and helpers
    def take_action(self, state_before, trial, hist, env):
        if self.__inside_option():
            values = self.v.get_option_values(state_before, self.option_stack[-1], self.theta)  # use option policy
        else:
            values = self.v.get()  # curiosity (self.v.c) + reward values (self.v.r)
        option = self.__select_option(values)
        self.once_per_option(hist, option, trial)
        if self.__is_basic(option):
            self.once_per_trial(trial, hist)
            return option  # execute option
        else:
            return self.take_action(state_before, trial, hist, env)  # use option policy to select next option(s)

    def __select_option(self, values):
        option = [1000, 1000]
        while option[0] > self.hier_level:  # make sure option comes from an allowed level
            if self.__is_greedy():
                selected_options = np.argwhere(values == np.nanmax(values))  # all options with the highest value
            else:
                selected_options = np.argwhere(~np.isnan(values))  # all options that are not nan
            select = np.random.randint(len(selected_options))  # randomly select the index of one of the options
            option = selected_options[select]  # pick that option
        return option

    def once_per_option(self, hist, option, trial):
        self.option_stack.append(option)
        self.v.step_counter[option[0], option[1]] = 0
        hist.action_s[trial, option[0]] = option[1]  # save action choice for every action selected in this trial

    def once_per_trial(self, trial, hist):
        hist.n[trial] = self.n.copy()
        hist.v[trial] = self.v.get()
        hist.r[trial] = self.v.r.copy()
        hist.c[trial] = self.v.c.copy()
        if self.hier_level > 0:
            hist.update_option_history(self, trial)

    # Learn and helpers
    def learn(self, hist, env, events, rewards, trial, state_before, state_after):
        self.__learn_from_events(hist, env, trial, events, state_before, state_after)  # count events, initialize new options (v & theta)
        self.__learn_rest(hist, trial, events, state_before, state_after)  # update theta of ongoing options, v of terminated
        self.v.learn_from_rewards(rewards, self.alpha)

    def __learn_from_events(self, hist, env, trial, events, state_before, state_after):
        current_events = np.argwhere(events)
        for event in current_events:
            self.n[event[0], event[1]] += 1  # update event count (novelty decreases)
            if self.__is_novel(event):
                self.v.create_option(event)
                self.theta.create_option(event, env, self.v.get())
                self.v.update_c(self, event, 1, events)  # update option value right away
            if not self.__is_basic(event):  # it's a higher-level event
                self.theta.update(self, hist, event, 1, state_before, state_after)  # update in-option policy
            hist.update_theta_history(self, event, trial)

    def __learn_rest(self, hist, trial, events, state_before, state_after):
        current_events = np.argwhere(events)
        for current_option in np.flipud(self.option_stack):  # go through options, starting at lowest-level one
            goal_achieved = self.__goal_achieved(current_option, current_events)
            if not self.__is_basic(current_option) and not goal_achieved:  # thetas not covered by learn_from_events
                hist.update_theta_history(self, current_option, trial)
                self.theta.update(self, hist, current_option, 0, state_before, state_after)
            if goal_achieved:  # goal achieved -> event happened -> update expected novelty toward perceived novelty
                if not self.__is_novel(current_option):  # novel events are already updated in learn_from_events
                    self.v.update_c(self, current_option, 1, events)
                self.finish_option()
            elif self.__is_distracted():  # goal not achieved -> event didn't happen -> update expected novelty toward 0
                self.v.update_c(self, current_option, 0, events)
                self.finish_option()
            else:  # if current_option has not terminated, no higher-level option can have terminated
                break  # no more updating needed

    def finish_option(self):
        self.option_stack.pop()
        if len(self.option_stack) > 0:
            new_option = self.option_stack[-1]
            self.v.step_counter[new_option[0], new_option[1]] += 1

    # Little helpers
    @staticmethod
    def __goal_achieved(option, events):
        return any([np.all(option == event) for event in events])  # did option's target event occur?

    def __is_distracted(self):
        return self.distraction > np.random.rand()

    def __is_greedy(self):
        return np.random.rand() > self.epsilon

    @staticmethod
    def __is_basic(option):
        return option[0] == 0

    def __is_novel(self, event):
        return self.n[event[0], event[1]] == 1

    def __inside_option(self):
        return len(self.option_stack) > 0
