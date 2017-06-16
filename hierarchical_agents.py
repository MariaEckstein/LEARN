import numpy as np


class HierarchicalAgent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, alpha, epsilon, distraction, n_levels, n_lights, n_lights_tuple):

        # Agent's RL features
        self.alpha = alpha
        self.epsilon = epsilon
        self.distraction = distraction

        # Characteristics of the environment
        self.n_levels = n_levels
        self.n_lights = n_lights
        self.n_lights_tuple = n_lights_tuple
        self.n_options = np.sum([n_lights // n_lights_tuple ** i for i in range(n_levels)])

        # Agent's thoughts about his options (values, novelty)
        self.v = np.random.rand(n_levels, n_lights)  # values of actions and options; same shape as state
        self.o_blocked = np.zeros([n_levels, n_lights], dtype=bool)  # INTEGRATE INTO agent.v LIKE FOR agent.o_v!
        self.o_blocked[1:] = 1  # initially, all options are blocked (basic actions in row 0 stay 0)
        self.n = np.zeros([n_levels, n_lights]).astype(np.int)  # counter for experience with events; same shape
        self.o_v = np.empty([self.n_options - n_lights + 1, n_lights])  # in-option values; one row per option
        row = 0
        for level in range(n_levels):
            n_options_level = n_lights // (2 * n_lights_tuple ** level)
            for option in range(n_options_level):
                self.o_v[row, :] = np.nan
                n_lights_level = n_lights // n_lights_tuple ** level  # number of lights at level below option
                self.o_v[row, range(n_lights_level)] = 0
                row += 1

        # Agent's memory about his plans and past actions
        self.o_list = []  # contains the o_v rows of each option that is currently being executed
        self.level = 0  # at which level are we in the option?
        self.v_in_charge = np.zeros(n_lights)
        self.actions = []

    def take_action(self, state, events):
        event_is = np.argwhere(events)
        # if option stack is empty, we choose among all actions and options
        if len(self.o_list) == 0:
            values = self.v.copy()
            values[self.o_blocked] = np.nan  # block out option that haven't been discovered yet
            [self.level, action] = self.select_action(values)
        # if we are inside an option, we choose an action according to in-option policy
        else:
            self.terminate_option(event_is)  # first check if we terminate option (reached sub-goal or got distracted)
            action = self.select_action(self.v_in_charge)
            # print("Just executed action", action, "inside option", self.o_list)
        # if selected action is not at basic level, we trickle down through the options until we reach the basic level
        if self.level > 0:
            print("Selected option", self.option_coord_to_index([self.level, action]).astype(int))
            action = self.trickle_down_hierarchy(action)
        self.actions.append(action)
        return action

    def trickle_down_hierarchy(self, action):
        while self.level > 0:
            print("Trickling down levels... now at level", self.level, "in option", self.o_list)
            option_index = self.option_coord_to_index([self.level, action]).astype(int)
            self.o_list.append(option_index)
            self.v_in_charge = self.o_v[self.o_list[-1], :]  # get the values of the last option in the stack
            self.level -= 1
            return self.select_action(self.v_in_charge)

    def learn(self, events):
        # if we encountered novel events, we create new options
        event_is = np.argwhere(events)
        for i in range(len(event_is)):
            self.create_option(event_is[i])
        # WE ALSO USE NOVELTY TO UPDATE BASIC ACTION VALUES
        RPE = len(event_is) - self.v[0, self.actions[-1]]
        self.v[0, self.actions[-1]] += self.alpha * RPE

    def create_option(self, event):
        option_index = self.option_coord_to_index(event).astype(int)
        # initialize option
        if self.o_blocked[event[0], event[1]]:
            self.o_blocked[event[0], event[1]] = False  # Un-block option
            self.v[event[0], event[1]] = 1  # initialize option value
            print("Option", option_index, "created!")
        # update in-option values
        self.o_v[option_index, self.actions[-1]] = 1  # ONLY WORKS FOR LEVEL-1 OPTIONS; HIGHER-LEVEL OPTIONS NEED TO UPDATE BASED ON OPTIONS, NOT self.actions!
        print("New in-option policy:", self.o_v[option_index, :])

    def terminate_option(self, event_is):
        # Case 1: We achieved the sub-goal
        option = self.o_list[-1]
        current_events = np.array([self.option_coord_to_index(event_is[i]) for i in range(len(event_is))])
        if np.any(current_events == option):
            self.resume_higher_option(option)
        # Case 2: We got distracted
        elif self.distraction > np.random.rand():
            print("Oooops, got distracted and quit option", option)
            self.o_list = []
            self.v_in_charge = self.v[0]  # SHOULD BE self.v, BUT THAT WOULDN'T WORK IN select_action
        # MISSING: UPDATE self.v!
            # Case 1: RPE according to novelty of sub-goal event
            # Case 2: RPE with 0 novelty

    def resume_higher_option(self, option):
        self.o_list.pop()  # get out of current option
        print("Sub-goal", option, "achieved! Now doing", self.o_list)
        if len(self.o_list) == 0:  # if we worked our way through the whole hierarchy of options, we're free again!
            self.v_in_charge = self.v[0]  # SHOULD BE self.v, BUT THAT WOULDN'T WORK IN select_action
            self.o_v[option, self.actions[-1]] = 1  # SHOULD WORK - update in-option values
            print("New in-option policy:", self.o_v[option, :])
        else:  # if we're still inside an option, put it in charge again
            self.v_in_charge = self.o_v[self.o_list[-1], :]
            self.level += 1  # we just moved one level up
            self.o_v[self.o_list[-1], option] = 1  # SHOULD WORK - update in-option values
            print("New in-option policy:", self.o_list[-1])

    def select_action(self, values):
        if np.random.rand() > self.epsilon:
            action = np.argwhere(values == np.nanmax(values))[0]
        else:
            available_actions = np.argwhere(~ np.isnan(values))
            select = np.random.choice(range(len(available_actions)))
            action = available_actions[select]
        return action

    def option_coord_to_index(self, coord):
        level, selected_a = coord
        if level == 0:
            n_options_below = 0
        else:
            n_tuples_level = self.n_lights // self.n_lights_tuple ** level
            n_options_below = np.sum([n_tuples_level * self.n_lights_tuple ** i for i in range(1, level)])
        return n_options_below + selected_a

# - select option
# - get in_option values
# - select actions according to option values until option terminates
# - when option terminates, take over values from the option in the level above
# - after termination, update value of the option according to novelty of resulting event
# - also update in_option values to learn how to achieve the event next time

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
        self.n[event] += 1  # Count how often each action has been performed
        event_novelty = 1 / self.n[action]
        if np.sum(event) > 0:
            self.n[2:] += event  # Count how often each higher-level event has occurred
            event_novelty = np.sum(1 / self.n[2:][event])
        else:
            event_novelty = 0
        return event_novelty + event_novelty
