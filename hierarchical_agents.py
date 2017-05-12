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
        available_v = self.get_values(state)
        action = self.select_action(available_v)
        self.level, selected_a = action
        if len(action) == 1 or self.level == 0:  # agent selected a basic action
            selected_a = action[-1]
            switch_to = 1 - state[0, selected_a]
            return switch_to, selected_a
        else:  # agent selected an option
            self.initiate_option(action, state)

    def get_values(self, state):
        # Get the values of the available actions / options
        if len(self.o_list) == 0:  # if option stack is empty, i.e., we have the choice among all actions and options
            available_v = self.v[1:].copy()
            available_v[0] = self.v[1-state[0], range(state.shape[1])]  # values of basic actions
            available_v[1:][self.o_blocked] = float('nan')  # block out option that haven't been discovered yet
            # block out options that aren't available right now (light already on) => NO!
        else:  # if the option stack is not empty, i.e., we are inside an option
            available_v = self.o_v[:, :, self.o_list.pop()]  # get the values of the last option in the stack
            if self.level == 0:  # if the option refers to basic actions
                available_v = available_v[1-state[0], range(state.shape[1])]
        return available_v

    def select_action(self, available_v):
        # Select an action / option, according to its value
        best_actions = np.argwhere(available_v == np.nanmax(available_v))
        worse_actions = np.argwhere(available_v < np.nanmax(available_v))
        if np.random.rand() > self.epsilon or len(worse_actions) == 0:
            select = np.random.choice(range(best_actions.shape[0]))
            action = best_actions[select]
        else:
            select = np.random.choice(range(worse_actions.shape[0]))
            action = worse_actions[select]
        return action

    def initiate_option(self, action, state):
        option_index = self.option_coord_to_index(action)
        self.o_list.append(option_index)
        self.level -= 1
        self.take_action(state)

    def option_coord_to_index(self, coord):
        level, selected_a = coord
        if level == 1:
            n_options_below = 0
        else:
            n_tuples_level = self.n_lights // self.n_lights_tuple ** level
            n_options_below = np.sum([n_tuples_level * self.n_lights_tuple ** i for i in range(1, level)])
        return n_options_below + selected_a


class OptionAgent(HierarchicalAgent):
    """
    This agent creates options.
    It is not driven by values and selects actions/options randomly.
    """

class NoveltyAgentH(HierarchicalAgent):
    """
    This agent is driven by novelty.
    It sees higher-level lights and recognizes their novelty (in addition to basic lights).
    It does not form options.
    """
    def update_values(self, old_state, action, new_state, high_lev_change):
        novelty = self.measure_novelty(action, high_lev_change)
        self.v[action] += self.alpha * (novelty - self.v[action])  # Does this already work for options?

    def measure_novelty(self, action, high_lev_change):
        self.n[action] += 1  # Count how often each basic action has been performed
        action_novelty = 1 / self.n[action]
        if np.sum(high_lev_change) > 0:
            self.n[2:] += high_lev_change  # Count how often each higher-level event has occurred
            event_novelty = np.sum(1 / self.n[2:][high_lev_change])
        else:
            event_novelty = 0
        return action_novelty + event_novelty

    def handle_options(self, high_lev_change, action):
        change_positions = np.argwhere(high_lev_change)
        for i in range(len(change_positions)):  # if more than one higher-level lights turned on during one move
            change_position = change_positions[i]
            coord = [change_position[0] + 1, change_position[1]]
            option_index = self.option_coord_to_index(coord)  # get index of option
            if self.o_blocked[change_position]:  # if the option doesn't exist yes (still blocked)
                self.create_option(change_position, option_index)
            # self.o_v[action, option_index] = 1  # initialize value of the previous, good action # DOESN'T WORK BECAUSE I WANT TO UPDATE THE VALUES OF OPTIONS, NOT JUST ACTIONS

    def create_option(self, change_position, option_index):
        self.o_blocked[change_position] = False  # Un-block the option
        self.o_v[:, :, option_index] = float('nan')  # block out all actions available to the option
        level = change_position[0]  # this is the level BELOW the option being created
        n_lights_level = self.n_lights // self.n_lights_tuple ** level  # number of lights at the level BELOW option
        if n_lights_level == self.n_lights:  # if option selects among basic actions
            self.o_v[:, :, option_index] = 0  # initialize two rows to 0
        else:  # if option selects among options
            self.o_v[0, range(n_lights_level), option_index] = 0
