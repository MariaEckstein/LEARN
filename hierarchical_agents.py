import numpy as np


class HierarchicalAgent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, alpha, epsilon, n_levels, n_lights, n_lights_tuple):
        # RL features
        self.alpha = alpha
        self.epsilon = epsilon
        # Features of the environment
        self.n_levels = n_levels
        self.n_lights = n_lights
        self.n_lights_tuple = n_lights_tuple
        self.n_options = np.sum([n_lights // n_lights_tuple ** i for i in range(n_levels)])
        # Agent's representation of the task (experienced novelty and action/option values)
        self.v = np.zeros([n_levels + 1, n_lights])  # values of actions and options; same organization as state array
        self.n = np.zeros([n_levels + 1, n_lights]).astype(np.int)  # counter for experience with events; same shape
        self.o_v = np.zeros([2, n_lights, self.n_options - n_lights + 1])  # inter-option policies; each option has one slice of dimension [1, n_lights] or [2, n_lights], corresponding to the values of the actions/options it can select
        self.o_blocked = np.ones([n_levels - 1, n_lights], dtype=bool)  # initially, all options are still blocked
        self.o_list = []  # contains the indexes of the slices in o_v of each option that is currently being executed
        self.level = 0  # where are we in the option?

    def take_action(self, state):

        # Get the values of the available actions / options
        if len(self.o_list) == 0:  # if option stack is empty, i.e., we have the choice among all actions and options
            available_v = self.v[1:].copy()
            available_v[0] = self.v[1-state[0], range(state.shape[1])]  # values of basic actions
            available_v[1:][self.o_blocked] = float('nan')  # block out option that haven't been discovered yet
        else:  # if the option stack is not empty, i.e., we are inside an option
            available_v = self.o_v[:, :, self.o_list.pop()]  # get the values of the last option in the stack
            if self.level == 0:  # if the option refers to basic actions
                available_v = available_v[1-state[0], range(state.shape[1])]

        # Select an action / option, according to its value
        best_actions = np.argwhere(available_v == np.nanmax(available_v))
        worse_actions = np.argwhere(available_v < np.nanmax(available_v))
        if np.random.rand() > self.epsilon or len(worse_actions) == 0:
            select = np.random.choice(range(best_actions.shape[0]))
            action = best_actions[select]
        else:
            select = np.random.choice(range(worse_actions.shape[0]))
            action = worse_actions[select]
        self.level, selected_a = action

        # Return the selected action / go into the selected option
        if len(action) == 1 or self.level == 0:  # agent selected a basic action
            selected_a = action[-1]
            switch_to = 1 - state[0, selected_a]
            return switch_to, selected_a
        else:  # agent selected an option
            if self.level == 1:
                n_options_below = 0
            else:
                n_tuples_level = self.n_lights // self.n_lights_tuple ** self.level
                n_options_below = np.sum([n_tuples_level * self.n_lights_tuple ** i for i in range(1, self.level)])
            selected_a = n_options_below + selected_a
            self.o_list.append(selected_a)
            self.level -= 1
            self.take_action(state)





            # selected_a = selected_a / self.n_lights_tuple ** level
            # if level == 1:
            #     n_options_below = 0
            # else:
            #     n_tuples_level = self.n_lights // self.n_lights_tuple ** level
            #     n_options_below = np.sum([n_tuples_level * self.n_lights_tuple ** i for i in range(1, level)]) - 1  # -1 because python indexing starts at 0
            # selected_o = selected_a + n_options_below
            # self.o_list.append(selected_o)  # option_i is the row number in self.o_v (0:1 for basic actions, etc.)
            # self.take_action(state, level - 1)

        # n_lights_tuple_l = self.n_lights_tuple ** level
        # n_tuples_level = self.n_lights // n_lights_tuple_l
        # value_indices = [n_lights_tuple_l * i for i in range(n_tuples_level)]
        # available_v = values[value_indices]

class OptionAgent(HierarchicalAgent):
    """
    This agent creates options.
    It is not driven by values and selects actions/options randomly.
    """
    def create_option(self, new_state, high_level_change):
        # Check that option doesn't exist yet
        # Initiation set: all states in which the lights are off
        # Termination set: all states in which the lights state[level, tuple] are on.
        # option values self.o_v: look back in time; credit past actions/option according to how recent they were


        # Make the option available
        self.o_blocked += high_level_change

        # Update the value of the option to 1
        # self.v[]

        # Update the within-option values



        # update_options()
        # update_action_values()
        # update_option_values()
        # train_options()
        # select_action()

    # def create_option(self, old_state, new_state):
    #     self.i_option += 1


class NoveltyAgentH(HierarchicalAgent):
    """
    This agent is driven by novelty.
    It sees higher-level lights and recognizes their novelty (in addition to basic lights).
    It does not form options.
    """
    def update_values(self, old_state, action, new_state, high_lev_change):
        novelty = self.measure_novelty(action, high_lev_change)
        self.v[action] += self.alpha * (novelty - self.v[action])  # Novelty instead reward

    def measure_novelty(self, action, high_lev_change):
        # Count how often each basic action has been performed
        self.n[action] += 1
        action_novelty = 1 / self.n[action]
        # Count how often each higher-level event has occured
        if np.sum(high_lev_change) > 0:
            self.n[2:] += high_lev_change
            event_novelty = np.sum(1 / self.n[2:][high_lev_change])
        else:
            event_novelty = 0
        return action_novelty + event_novelty
