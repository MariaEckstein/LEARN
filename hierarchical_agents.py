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

        # Agent's thoughts about its environment (values, novelty)
        self.initial_value = 0.5
        self.n = np.zeros([n_levels, n_lights])  # event counter
        self.n[1:] = np.nan
        self.v = self.initial_value * np.ones([n_levels, n_lights])  # feature weights (thetas) for actions and options
        self.v[1:] = np.nan  # undefined for options
        self.o_theta = np.full([self.n_options - n_lights + 1, n_lights, n_lights], np.nan)  # in-option features
        row_ot = 0
        for level in range(n_levels):
            n_options_level = n_lights // (2 * n_lights_tuple ** level)
            for option in range(n_options_level):
                n_lights_level = n_lights // n_lights_tuple ** level  # number of lights at level below option
                self.o_theta[row_ot, range(n_lights_level), :] = self.initial_value
                row_ot += 1

        # Agent's memory about his plans and past actions
        self.option_stack = []  # stack of the option(s) that are currently in charge
        self.option_selected_in_this_trial = None
        self.level = 0  # at which level are we in the option?
        self.theta_in_charge = self.o_theta[0, 0, :].copy()
        self.options = [list() for _ in range(n_levels)]  # options taken; one list per level
        [self.goal_achieved, self.distracted] = [False, False]

    def take_action(self, old_state):
        values = self.get_values(old_state)  # use v if not inside an option and theta if inside an option
        option = self.select_option(values)
        if len(self.option_stack) == 0:  # not inside an option
            print("Had free choice and selected option", option)
            self.level = option[0]  # level is determined by option selection
        else:  # inside an option
            print("Inside option(s)", self.option_stack)
            option[0] = self.level  # artifact of how I coded it -> level is set the first time the option is selected
        self.option_selected_in_this_trial = option
        action = self.get_basic_action(option, old_state)  # trickle down hierarchy until we reach basic level
        print("Option:", option, "Action:", action, "Option history:", self.options)
        return action

    def get_values(self, old_state):
        if len(self.option_stack) == 0:  # we're not inside an option -> basic values for option selection
            values = self.v.copy()
        else:  # we're inside an option -> follow in-option policy based on features
            features = 1 - old_state[self.level]  # features indicate which lights are OFF
            values = np.dot(self.theta_in_charge, features)  # calculate values from thetas
            nans = np.full([1, len(values)], np.nan)  # add another row to make values same shape as above
            values = np.vstack([values, nans])
        return values

    def select_option(self, values):
        if np.random.rand() > self.epsilon:
            option = np.argwhere(values == np.nanmax(values))[0]
        else:
            available_options = np.argwhere(~ np.isnan(values))
            select = np.random.choice(range(len(available_options)))
            option = available_options[select]
        return option

    def get_basic_action(self, option, old_state):
        self.options[self.level].append(option[1])  # remember we chose this option
        if self.level == 0:  # we selected a basic action -> return it right away
            return option
        else:  # we selected an option -> use it to select an option / action at the level below
            print("Trickling down levels - starting in option", option)
            self.option_stack.append(self.option_coord_to_index(option))  # put this option on top of the option stack
            self.level -= 1  # move down one level
            self.theta_in_charge = self.o_theta[self.option_stack[-1], :]  # get the thetas of this option
            values = self.get_values(old_state)  # get values from in-option thetas
            option = [self.level, self.select_option(values)[1]]  # select a new option according to values
            print("Option:", self.option_stack[-1], "; theta_in_charge:", np.round(self.theta_in_charge, 2), "; level:", self.level)
            return self.get_basic_action(option, old_state)  # repeat until we get a basic action

    def learn(self, old_state, events, action):
        # self.n[action[0], action[1]] += 1  # NOT NECESSARY ANY MORE BECAUSE EVENTS INCLUDES BASIC EVENTS NOW! novelty of basic event decreases
        current_events = np.argwhere(events)
        for event in current_events:
            self.n[event[0], event[1]] += 1  # novelty of event decreases
            self.create_option(old_state, event)  # create option for (higher-level) events that haven't been discovered
        for option in self.option_stack.append(action):  # DOESN'T WORK OBVIOUSLY
            [goal_achieved, distracted] = self.check_if_goal_achieved_distracted(option)
            if goal_achieved or distracted:  # option terminated -> update v and in-option thetas
                self.update_option_v_theta(old_state, events)
                print("For option", option, "goal was achieved:", goal_achieved, "; got distracted:", distracted)
            else:  # option did not terminate -> no learning
                print("Option", option, "did not terminate; option stack:", self.option_stack, "; no learning.")

        # if len(self.option_stack) == 0:  # option terminated -> update v and in-option theta!
        #     print("Learning outside options (len(self.option_stack) == 0).")
        #     self.update_v(old_state)  # update agent.v according to last action and outcome
        #     self.theta_in_charge = self.v.copy()
        # else:
        #     self.update_theta(old_state, events)  # if option ends, update_ot & update_v & update theta_in_charge

    def update_option_v_theta(self, old_state, events):
            self.update_v(old_state)
            self.update_ot(self.goal_achieved, old_state)
            self.option_stack.pop()  # check off current option
            if len(self.option_stack) == 0:
                self.theta_in_charge = self.v.copy()
            else:
                self.theta_in_charge = self.o_theta[self.option_stack[-1], :]
                self.level += 1
                return self.update_option_v_theta(old_state, events)

    def update_v(self, old_state):


        print("Previous v:", np.round(self.v, 2))
        print("N:", self.n)
        if len(self.options[self.level]) > 0:  # we have previously executed an action/option at this level
            if len(self.option_stack) == 0:  # we're not inside an option -> update basic actions
                index = [self.level, self.options[self.level][-1]]
                print("Updating action", index)
            else:  # we are inside an option -> update options
                index = [self.level+1, self.options[self.level+1][-1]]
                print("Updating option", index)
            novelty = 1 / self.n[index[0], index[1]]  # SHOULD BE TOTAL NOVELTY OF TRIAL, I.E., LEN(EVENTS)?
            self.rescorla_wagner(old_state, index, novelty, self.v)
        print("New theta:", np.round(self.v, 2))

    def check_if_goal_achieved_distracted(self, events):  # NEEDS REWORKING! NEEDS TO WORK FOR BASIC ACTIONS TOO, WHICH DON'T HAVE OPTION_COORD!
        current_events = np.argwhere(events)
        # basic_events = ...
        # if len(basic_events) > 0:
            # goal_achieved = True
            # remove basic_events from events
        current_events_i = np.array([self.option_coord_to_index(current_events[i]) for i in range(len(current_events))])
        goal_achieved = np.any(current_events_i == self.option_stack[-1])
        distracted = self.distraction > np.random.rand()
        return [goal_achieved, distracted]

    def update_theta(self, old_state, events):
        [self.goal_achieved, self.distracted] = self.check_if_goal_achieved_distracted(events)
        if self.goal_achieved or self.distracted:
            self.update_ot(self.goal_achieved, old_state)
            self.update_v(old_state)
            self.option_stack.pop()  # check off current option
            print("Goal achieved:", self.goal_achieved, "; got distracted:", self.distracted, "; will now do", self.option_stack)
            if len(self.option_stack) == 0:
                self.theta_in_charge = self.v.copy()
            else:
                self.theta_in_charge = self.o_theta[self.option_stack[-1], :]
                self.level += 1
                return self.update_theta(old_state, events)
        else:
            print("Still inside option(s)", self.option_stack, ", so no learning.")

    def create_option(self, old_state, event):
        if np.any(np.isnan(self.v[event[0], event[1]])):  # if option has not yet been discovered
            option_index = self.option_coord_to_index(event)
            # Initialize value of the new option
            self.v[event[0], event[1]] = self.initial_value  # initialize value
            RPE = self.n[event[0], event[1]] - self.v[event[0], event[1]]  # = 1 - initial_value
            self.v[event[0], event[1]] += self.alpha * RPE  # and update value
            # Learn in-option policy of the new option
            features = 1 - old_state[(event[0] - 1)]  # which lights are on and off at the level below the event?
            action_that_led_to_this_event = self.options[event[0]-1][-1]  # which action/option led to the event?
            theta = self.o_theta[option_index, action_that_led_to_this_event, :]  # feature weights of this option
            value = np.dot(features, theta)  # previous value of this option
            RPE = 1 - value
            self.o_theta[option_index, action_that_led_to_this_event, :] += self.alpha * RPE * features
            # Info
            print("state[", (event[0]-1), "]:", old_state[(event[0] - 1)])
            print("Option", option_index, event, "with v[", event[0], event[1], "]",
                  self.v[event[0], event[1]], "created!")
            print("Option", option_index, event, "has o_theta[", option_index, action_that_led_to_this_event,
                  ", :]", self.o_theta[option_index, action_that_led_to_this_event, :])

    def update_ot(self, goal_achieved, old_state):
        previous_option = [self.level, self.options[self.level][-1]]
        print("Previous ot of option", self.option_stack[-1], ":", np.round(self.o_theta[self.option_stack[-1], :], 2))
        index = [self.option_stack[-1], previous_option[1]]
        self.rescorla_wagner(old_state, index, goal_achieved, self.o_theta)
        print("New ot of option", self.option_stack[-1], ":", np.round(self.o_theta[self.option_stack[-1], :], 2))

    def rescorla_wagner(self, old_state, index, reward, theta):
        RPE = reward - theta[index[0], index[1], :]
        features = 1 - old_state[self.level]  # features indicate which lights are OFF; state indicates which are ON
        theta[index[0], index[1], :] += self.alpha * RPE * features  # SIDE EFFECTS!

    def option_coord_to_index(self, coord):
        level, selected_a = coord
        if level == 0:
            n_options_below = 0
        else:
            n_tuples_level = self.n_lights // self.n_lights_tuple ** level
            n_options_below = np.sum([n_tuples_level * self.n_lights_tuple ** i for i in range(1, level)])
        index = n_options_below + selected_a
        return int(index)

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
