import numpy as np

class HierarchicalAgent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, alpha, epsilon, n_levels, n_lights, n_lights_tuple):
        self.alpha = alpha
        self.epsilon = epsilon
        self.n_options = np.sum([n_lights // n_lights_tuple ** i for i in range(n_levels)])
        self.v = np.zeros([n_levels + 1, n_lights])  # values of actions and options; same organization as state array
        self.n = np.zeros([n_levels + 1, n_lights]).astype(np.int)  # counter for experience with events; same shape
        self.o_v = np.zeros([self.n_options - n_lights + 1, n_lights])  # inter-option policies; each option has one row, corresponding to the values of the options it can select
        self.o_mask = np.zeros([n_levels - 1, n_lights]).astype(np.int)
        self.o_list = []

    def take_action(self, state):
        if not self.o_v:  # when the list is empty
            values = self.v
        else:
            values = self.o_v[self.o_list[-1]]  # get the values of the current option

        if values.shape[0] == 2:  # actions
            available_values = values[1-state[0], range(len(state[0]))]
            best_actions = np.argwhere(available_values == np.max(available_values)).flatten()
            worse_actions = np.argwhere(available_values < np.max(available_values)).flatten()
            if (np.random.rand() > self.epsilon) or (len(worse_actions) == 0):
                light_i = np.random.choice(best_actions)
            else:
                light_i = np.random.choice(worse_actions)
            switch_to = 1 - state[0, light_i]
            return light_i, switch_to
        else:  # options
            best_actions = np.argwhere(values == np.max(values)).flatten()
            worse_actions = np.argwhere(values < np.max(values)).flatten()
            if (np.random.rand() > self.epsilon) or (len(worse_actions) == 0):
                option_i = np.random.choice(best_actions)
            else:
                option_i = np.random.choice(worse_actions)
            self.o_list.append(option_i)  # option_i is the row number in self.o_v (0:1 for basic actions, etc.)
            return self.take_action(state)


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
        self.o_mask += high_level_change

        # Update the value of the option to 1
        self.v[]

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
    It perceives higher-level lights and perceives their novelty in addition to basic lights.
    It does not form options.
    """
    def update_values(self, old_state, action, new_state, high_lev_change):
        light_i, switch_to = action
        novelty = self.measure_novelty(action, high_lev_change)
        self.v[switch_to, light_i] += self.alpha * (novelty - self.v[switch_to, light_i])  # Novelty instead reward

    def measure_novelty(self, action, high_lev_change):
        # Count how often each basic action has been performed / experienced
        light_i, switch_to = action  # switch_to == 0: switch light off; switch_to == 1: switch light on
        self.n[switch_to, light_i] += 1  # update rows 0 (switch light off) and 1 (switch on)
        action_novelty = 1 / self.n[switch_to, light_i]
        # Count how often each higher-level event has been experienced
        if np.sum(high_lev_change) > 0:
            self.n[2:] += high_lev_change
            event_novelty = 1 / self.n[2:][high_lev_change == 1][0]
        else:
            event_novelty = 0
        return action_novelty + event_novelty
