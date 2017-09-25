import numpy as np
from theta import Theta
from v import V


# TDs:
# - options of different forms (different numbers of actions, overlapping)
# - decide whether lights will stay on (problem humans can use turning-off lights as cues as to which lights belong together in an option) or turn off (problem: model has infinite memory, people not)
# - find a better way to select actions, not going from left to right
# - make sure everything is working
# - what is the agent's goal? Just unstructured exploration? Figure it out? Turn on many lights? Try to find new colors?
# - look at behavior in more detail
# - compare agents
# - bring to humans
# - add WM: humans can remember that they were executing an option and can give credit to it even when they gave up,
#   once they see the event occur / the sub-goal reached
# - add eligibility traces -> backward view -> people recall multiple actions they took before an event
# - humans know that each light belongs to only one group -> reduce values of lights with high values in another group?

class HierarchicalAgent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, alpha, epsilon, distraction, env):

        # Agent's RL features
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # greediness
        self.distraction = distraction  # propensity to quit unfinished options

        # Agent's thoughts about its environment (novelty, values, theta)
        self.n = np.zeros([env.n_levels, env.n_lights])  # event counter = inverse novelty
        self.trial = 0  # current trial
        self.v = V(env)
        self.theta = Theta(env)

        # Agent's plans and memory about his past actions
        self.option_stack = []  # stack of the option(s) that are currently guiding behavior
        self.option_history = np.zeros([env.n_trials * env.n_levels, env.n_levels, env.n_lights + 2])
        self.option_row = 0
        self.action_history = np.zeros([env.n_trials, env.n_lights])

    def take_action(self, old_state):
        self.v.history[self.trial, :, :] = self.v.get()
        values = self.v.get_option_values(old_state, self.option_stack, self.theta)  # self.__get_option_values(old_state)
        option = self.__select_option(values)
        self.option_stack.append(option)
        if self.__is_basic(option):
            return option  # execute option
        else:
            return self.take_action(old_state)  # use option policy to select next option(s)

    def __select_option(self, values):
        if self.__is_greedy():
            selected_options = np.argwhere(values == np.nanmax(values))  # all options with the highest value
        else:
            selected_options = np.argwhere(~ np.isnan(values))  # all options that are not nan
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

    def learn(self, old_state, events):
        self.__update_theta_history(self.trial)
        self.__update_option_history()
        current_events = np.argwhere(events)
        self.__learn_from_events(old_state, current_events)  # count events, initialize new options (v & theta)
        self.__learn_rest(old_state, current_events, [])  # update theta of ongoing options, v of terminated

    def __learn_from_events(self, old_state, current_events):
        previous_option = []
        for event in current_events:
            self.n[event[0], event[1]] += 1  # update event count (novelty decreases)
            if self.__is_novel(event):
                self.v.create_option(event)
                self.v.update(self, event, 1)  # update option value right away
            if not self.__is_basic(event):  # it's a higher-level event
                self.theta.update(event, 1, old_state, previous_option, self)  # update in-option policy
            previous_option = event.copy()

    def __update_theta_history(self, trial):
        self.theta.history[self.theta.h_row, :, :, :self.theta.n_lights] = self.theta.get()
        self.theta.history[self.theta.h_row, :, :, -1] = trial
        self.theta.h_row += 1

    def __update_option_history(self):
        self.step = 0
        for current_option in self.option_stack:  # list all current options
            self.option_history[self.option_row, current_option[0], current_option[1]] = 1
            self.option_history[self.option_row, :, -2] = self.trial
            self.option_history[self.option_row, :, -1] = self.step
            self.step += 1
            self.option_row += 1

    def __learn_rest(self, old_state, events, previous_option):
        for current_option in np.flipud(self.option_stack):  # go through options, starting at lowest-level one
            [goal_achieved, distracted] = self.__get_goal_achieved_distracted(current_option, events)
            if not self.__is_basic(current_option) and not goal_achieved:  # thetas not covered by learn_from_events
                self.theta.update(current_option, 0, old_state, previous_option, self)
            if goal_achieved:  # goal achieved -> event happened -> update expected novelty toward perceived novelty
                if not self.__is_novel(current_option):  # novel events are already updated in learn_from_events
                    self.v.update(self, current_option, 1)
                previous_option = self.option_stack.pop()
            elif distracted:  # goal not achieved -> event didn't happen -> update expected novelty toward 0
                self.v.update(self, current_option, 0)
                previous_option = self.option_stack.pop()
            else:  # if current_option has not terminated, no higher-level option can have terminated
                break  # no more updating needed

    def __get_goal_achieved_distracted(self, option, events):
        goal_achieved = any([np.all(option == event) for event in events])  # did option's target event occur?
        distracted = self.distraction > np.random.rand()
        return [goal_achieved, distracted]


class OptionAgent(HierarchicalAgent):
    """
    This agent creates options.
    It is not driven by values and selects actions/options randomly.
    """


class NoveltyAgentH(HierarchicalAgent):
    """
    This agent is driven by novelty.
    It sees higher-level lights and recognizes their novelty (in addition to basic lights).
    It does also form options.
    """

