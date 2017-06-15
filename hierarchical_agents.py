import numpy as np


class HierarchicalAgent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, alpha, epsilon, distraction, n_levels, n_lights, n_lights_tuple):
        # RL features
        self.alpha = alpha
        self.epsilon = epsilon
        self.distraction = distraction
        # Features of the environment
        self.n_levels = n_levels
        self.n_lights = n_lights
        self.n_lights_tuple = n_lights_tuple
        self.n_options = np.sum([n_lights // n_lights_tuple ** i for i in range(n_levels)])
        # Agent's representation of the task (experienced novelty and action/option values)
        self.v = np.random.rand(n_levels, n_lights)  # values of actions and options; same organization as state array
        self.n = np.zeros([n_levels, n_lights]).astype(np.int)  # counter for experience with events; same shape
        self.o_v = np.zeros([self.n_options - n_lights + 1, n_lights])  # COULD I INITIALIZE THE WHOLE THING AS NAN? inter-option policies; each option has one slice of dimension [1, n_lights] or [2, n_lights], corresponding to the values of the actions/options it can select
        self.o_blocked = np.ones([n_levels - 1, n_lights], dtype=bool)  # initially, all options are blocked
        self.o_list = []  # contains the indexes of the slices in o_v of each option that is currently being executed
        self.level = 0  # where are we in the option?
        self.v_in_charge = np.zeros(n_lights)

    def take_action(self, state):
        if len(self.o_list) == 0:  # if option stack is empty, we can choose among all actions and options
            values = self.v.copy()
            values[1:][self.o_blocked] = np.nan  # block out option that haven't been discovered yet
            [self.level, action] = self.select_action(values)
        else:  # if we are inside an option
            self.terminate_option()  # terminate option if we reached the sub-goal (or got distracted)
            action = self.select_action(self.v_in_charge)

        while self.level > 0:  # now, trickle down through the options until we reach a basic action
            option_index = self.option_coord_to_index(action)
            self.o_list.append(option_index)
            self.v_in_charge = self.o_v[self.o_list[-1], :]  # get the values of the last option in the stack
            action = self.select_action(self.v_in_charge)
            self.level -= 1

        return action

    def terminate_option(self):
        # NOT SURE IF THAT WORKS! #
        termination_event = self.o_list[-1]
        events = np.argwhere(environment.event)
        current_events = []
        for i in range(len(events)):  # if more than one higher-level lights turned on during one move
            level, light_i = events[i]
            coord = [level + 1, light_i]
            option_index = self.option_coord_to_index(coord)  # get index of option
            current_events.append(option_index)
        if np.any(np.array(current_events) == termination_event) or self.distraction > np.random.rand():
            self.o_list.pop()
            if len(self.o_list) == 0:
                self.v_in_charge = self.v[0]
            else:
                self.v_in_charge = self.o_v[self.o_list[-1], :]
                self.level += 1  # we just moved one level up
            # update values
            # STILL MISSING!

    def select_action(self, values):
        if np.random.rand() > self.epsilon:
            action = np.argwhere(values == np.nanmax(values))[0]
        else:
            # DEBUG!!! SHOULD TEST WHETHER IT'S NAN, NOT > 0!!
            av_actions = values[0]
            select = np.random.choice(range(len(av_actions)))
            action = av_actions[select]

        return action

    def option_coord_to_index(self, coord):
        level, selected_a = coord
        if level == 1:
            n_options_below = 0
        else:
            n_tuples_level = self.n_lights // self.n_lights_tuple ** level
            n_options_below = np.sum([n_tuples_level * self.n_lights_tuple ** i for i in range(1, level)])
        return n_options_below + selected_a

    def handle_options(self, event, action):
        events = np.argwhere(event)
        for i in range(len(events)):  # if more than one higher-level lights turned on during one move
            level, light_i = events[i]
            coord = [level + 1, light_i]
            option_index = self.option_coord_to_index(coord)  # get index of option
            if self.o_blocked[level, light_i]:  # if the option doesn't exist yet (still blocked)
                self.create_option([level, light_i], option_index)
            if level == 0:
                self.o_v[action[0], action[1], option_index] = 1  # initialize value of the previous, good action # DOESN'T WORK BECAUSE I WANT TO UPDATE THE VALUES OF OPTIONS, NOT JUST ACTIONS
            else:
                self.o_v[0, action[1] // self.n_lights_tuple ** level, option_index] = 1

    def create_option(self, change_position, option_index):
        self.o_blocked[change_position] = False  # Un-block the option
        self.v[change_position[0] + 2, change_position[1]] = 1  # initialize option value at 1
        self.o_v[option_index, :] = np.nan  # block out all actions
        level = change_position[0]  # this is the level BELOW the option being created
        n_lights_level = self.n_lights // self.n_lights_tuple ** level  # number of lights at the level BELOW option
        self.o_v[option_index, range(n_lights_level)] = 0

# - select option
# - get in_option values
# - select actions according to option values until option terminates
# - when option terminates, take over values from the option in the level above
# - after termination, update value of the option according to novelty of resulting event
# - also update in_option values to learn how to achieve the event next time


    #     if len(self.o_list) == 0:  # if option stack is empty, i.e., we have the choice among all actions and options
    #         values = self.v.copy()
    #         values[1:][self.o_blocked] = float('nan')  # block out option that haven't been discovered yet
    #         action = self.select_action(values)
    #         self.level = action[0]
    #         while self.level > 0:  # if agent selected option, trickle down through options until we reach basic action
    #             option_index = self.option_coord_to_index(action)
    #             self.o_list.append(option_index)
    #             values = self.o_v[:, :, self.o_list[-1]]  # get the values of the last option in the stack
    #             action = self.select_action(values)
    #             self.level -= 1
    #         return action
    #
    #     while len(self.o_list) > 0:
    #         self.execute_option()
    #
    # def execute_option(self):
    #     values = self.o_v[:, :, self.o_list[-1]]  # get the values of the last option in the stack
    #     action = self.select_action(values)
    #     return action


    # def get_values(self):
    #     if len(self.o_list) == 0:  # if option stack is empty, i.e., we have the choice among all actions and options
    #         values = self.v.copy()
    #         values[1:][self.o_blocked] = float('nan')  # block out option that haven't been discovered yet
    #         action = self.select_action(values)
    #         self.level = action[0]
    #     else:  # if the option stack is not empty, i.e., we are inside an option
    #         values = self.o_v[:, :, self.o_list[-1]]  # get the values of the last option in the stack
    #         if self.level == 0:  # if the option refers to basic actions
    #             values = self.v[0]
    #         action = self.select_action(values)
    #         self.level = action[0]
    #     return values

    # def initiate_option(self, action, state):
    #     option_index = self.option_coord_to_index(action)
    #     self.o_list.append(option_index)
    #     self.level -= 1
    #     self.take_action(state)


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
    def update_values(self, old_state, action, new_state, event):
        novelty = self.measure_novelty(action, event)
        # if np.sum(event) > 0:
        #     events = np.argwhere(event)
        #     for i in range(len(events)):  # if more than one higher-level lights turned on during one move
        #         level, light_i = events[i]
        self.v[action] += self.alpha * (novelty - self.v[action])  # ALSO SHOULD INCORPORATE REWARDS / NOVELTY ACCRUED DURING OPTION EXECUTION

    def measure_novelty(self, action, event):
        self.n[action] += 1  # Count how often each basic action has been performed
        action_novelty = 1 / self.n[action]
        if np.sum(event) > 0:
            self.n[2:] += event  # Count how often each higher-level event has occurred
            event_novelty = np.sum(1 / self.n[2:][event])
        else:
            event_novelty = 0
        return action_novelty + event_novelty
