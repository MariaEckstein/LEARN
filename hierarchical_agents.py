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

        # Agent's thoughts about his environment (values, novelty)
        self.n = np.zeros([n_levels, n_lights])  # counter for experience with events; same shape
        self.n[1:] = np.nan
        self.v = 0 * np.ones([n_levels, n_lights])  # values of actions and options; same shape as state
        self.v[1:] = np.nan
        self.o_v = np.full([self.n_options - n_lights + 1, n_lights], np.nan)  # in-option values; one row per option
        row_ov = 0
        for level in range(n_levels):
            n_options_level = n_lights // (2 * n_lights_tuple ** level)
            for option in range(n_options_level):
                n_lights_level = n_lights // n_lights_tuple ** level  # number of lights at level below option
                self.o_v[row_ov, range(n_lights_level)] = 0
                row_ov += 1

        # Agent's memory about his plans and past actions
        self.option_stack = []  # contains the o_v rows of each option that is currently being executed
        self.level = 0  # at which level are we in the option?
        self.v_in_charge = np.zeros(n_lights)  # that's not the shape it will have
        self.options = [list() for _ in range(n_levels)]  # options taken

    def take_action(self, state, events):
        # novelty = 0.5 * len(np.argwhere(events))  # TD: NEEDS TO BE AGENT'S NOVELTY DETECTION!!
        print("Events:", np.argwhere(events))
        self.create_options(events)  # if an event happened, create the corresponding option
        if len(self.option_stack) == 0:  # we're not inside an option -> choose among all the possibilities
            self.update_v()  # update agent.v according to last action and outcome
            self.v_in_charge = self.v.copy()
            option = self.select_option()
            self.level = option[0]
            print("Had all the choice and selected option", option)
        else:  # we're inside an option -> follow option policy
            print("Inside option", self.option_stack)
            option = self.follow_option(events)  # update o_v; check if option terminates, if yes update v
        print("Option:", option)
        action = self.get_basic_action(option)  # trickle down hierarchy until we reach basic level
        self.n[action[0], action[1]] += 1
        print("Action:", action)
        self.options[self.level].append(action[1])
        print("Agent.options:", self.options)
        return action

    def create_options(self, events):
        current_events = np.argwhere(events)
        for event in current_events:
            if np.isnan(self.v[event[0], event[1]]):  # if option has not yet been discovered
                self.v[event[0], event[1]] = 1  # initialize it
                self.n[event[0], event[1]] = 1
                print("Option", self.option_coord_to_index(event), event, "with value", 1, "created!")
            else:
                self.n[event[0], event[1]] += 1
            # Not sure how to update in-option policies... I can't just use the last action because that's always
            # a basic action, and I don't know which one was the last higher-level option that I chose...
            # RPE = 1 - self.o_v[event_index, self.options[self.level][-1]]  # ????????????
            # self.o_v[event_index, self.options[self.level][-1]] += self.alpha * RPE  # ?????????????

    def follow_option(self, events):
        [goal_achieved, distracted] = self.check_if_goal_achieved_distracted(events)
        self.update_ov(goal_achieved)
        if goal_achieved or distracted:
            self.update_v()
            self.option_stack.pop()  # check off current option
            print("Goal achieved:", goal_achieved, "; got distracted:", distracted, "Will now do", self.option_stack)
            if len(self.option_stack) == 0:
                self.v_in_charge = self.v.copy()
                return [self.level, self.select_option()[1]]
            else:
                self.v_in_charge = self.o_v[self.option_stack[-1], :]
                self.level += 1
                return self.follow_option(events)
        else:
            return [self.level, self.select_option()[1]]

    def check_if_goal_achieved_distracted(self, events):
        current_events = np.argwhere(events)
        current_events_i = np.array([self.option_coord_to_index(current_events[i]) for i in range(len(current_events))])
        goal_achieved = np.any(current_events_i == self.option_stack[-1])
        distracted = self.distraction > np.random.rand()
        return [goal_achieved, distracted]

    def update_ov(self, goal_achieved):
        previous_option = [self.level, self.options[self.level][-1]]
        print("Previous ov of option", self.option_stack[-1], ":", self.o_v[self.option_stack[-1], :])
        RPE = goal_achieved - self.o_v[self.option_stack[-1], previous_option[1]]
        self.o_v[self.option_stack[-1], previous_option[1]] += self.alpha * RPE
        print("New ov of option", self.option_stack[-1], ":", self.o_v[self.option_stack[-1], :])

    def update_v(self):
        print("Previous v:", self.v)
        print("N:", self.n)
        if len(self.options[self.level]) > 0:
            if len(self.option_stack) == 0:  # we're not inside an option -> update basic actions
                index = [self.level, self.options[self.level][-1]]
                print("Updating action", [self.level, self.options[self.level][-1]])
            else:  # we are inside an option -> update options
                index = [self.level+1, self.options[self.level+1][-1]]
                print("Updating option", index)
            novelty = 1 / self.n[index[0], index[1]]
            RPE = novelty - self.v[index[0], index[1]]
            self.v[index[0], index[1]] += self.alpha * RPE
        print("New v:", self.v)

    def select_option(self):
        values = self.v_in_charge
        if len(values.shape) == 1:  # values is just 1 row
            nans = np.full([1, len(values)], np.nan)
            values = np.vstack([values, nans])
        if np.random.rand() > self.epsilon:
            option = np.argwhere(values == np.nanmax(values))[0]
        else:
            available_options = np.argwhere(~ np.isnan(values))
            select = np.random.choice(range(len(available_options)))
            option = available_options[select]
        return option

    def get_basic_action(self, option):
        if self.level == 0:
            return option
        else:
            print("Trickling down levels... now in option", self.option_stack, option)
            self.options[self.level].append(option[1])
            option_i = self.option_coord_to_index(option).astype(int)
            self.option_stack.append(option_i)
            self.level -= 1
            self.v_in_charge = self.o_v[self.option_stack[-1], :]  # get the values of the new option
            print("Option", self.option_stack[-1], "v_in_charge:", self.v_in_charge)
            option = [self.level, self.select_option()[1]]
            print(self.level)
            return self.get_basic_action(option)

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

    def measure_novelty(self, action, event):
        self.n[event] += 1  # Count how often each action has been performed
        event_novelty = 1 / self.n[action]
        if np.sum(event) > 0:
            self.n[2:] += event  # Count how often each higher-level event has occurred
            event_novelty = np.sum(1 / self.n[2:][event])
        else:
            event_novelty = 0
        return event_novelty + event_novelty
