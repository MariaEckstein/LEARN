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
        self.theta_in_charge = []
        [self.goal_achieved, self.distracted] = [False, False]
        self.history = []

    def take_action(self, old_state):
        # Get values
        if len(self.option_stack) == 0:  # not inside an option -> select option based on agent.v
            values = self.v.copy()
        else:  # inside an option -> select option based on in-option policy
            values = self.get_option_values(self.option_stack[-1], old_state)
        # Select (higher-level) option
        option = self.select_option(values)
        # Find basic-level action to execute
        if option[0] == 0:  # level == 0 means we selected a basic action -> return it right away
            return option
        else:  # level > 0 means we selected a higher-level option -> use its policy for option selection
            self.theta_in_charge = self.o_theta[self.option_stack[-1], :, :]  # get the thetas of this option
            print("Trickling down levels - starting in option", option)
            return self.take_action(old_state)

    def get_option_values(self, option, old_state):
        values = np.full([self.n_levels, self.n_lights], np.nan)  # initialize value array
        features = 1 - old_state[option[0]-1]  # features indicate which lights are OFF
        option_values = np.dot(self.theta_in_charge, features)  # calculate values from thetas
        values[option[0]-1, :] = option_values
        return values

    def select_option(self, values):
        if np.random.rand() > self.epsilon:
            option = np.argwhere(values == np.nanmax(values))[0]
        else:
            available_options = np.argwhere(~ np.isnan(values))
            select = np.random.choice(range(len(available_options)))
            option = available_options[select]
        self.option_stack.append(option)  # MAKE IT WORK FOR BASIC ACTIONS! put this option on top of the option stack
        self.history.append(option)  # for data analysis / debugging
        return option

    def learn(self, old_state, events):
        # Handle events: update novelty & create options
        current_events = np.argwhere(events)
        for event in current_events:
            self.n[event[0], event[1]] += 1  # novelty of event decreases
            if np.any(np.isnan(self.v[event[0], event[1]])):  # if option has not yet been discovered
                self.create_option(event)  # create option
        # Learn about terminated options: update v & theta
        self.update_terminated_options(current_events, old_state)

    def update_terminated_options(self, events, old_state):
        if len(self.option_stack) == 0:  # all options have terminated -> done
            self.theta_in_charge = []  # not necessary but will lead to errors if there's a bug -> debugging
            return []
        else:  # there are options in the stack -> see if the lowest-level one has terminated
            option = self.option_stack[-1]
            [goal_achieved, distracted] = self.check_if_goal_achieved_distracted(option, events)  # SHOULD ALSO BE ABLE TO GET DISTRACTED IN HIGHER-LEVEL OPTIONS!
            if goal_achieved or distracted:  # option terminated -> update v and in-option thetas
                self.option_stack.pop()
                self.update_v(option, goal_achieved)
                if option[0] > 0:
                    self.update_theta(option, goal_achieved, old_state)  # ONLY FOR HIGHER-ORDER OPTIONS!
                    self.theta_in_charge = self.o_theta[self.option_coord_to_index(option), :, :]
                    print("For option", option, "goal was achieved:", goal_achieved, "; got distracted:", distracted,
                          "new theta_in_charge:", self.o_theta[self.option_coord_to_index(option), :, :])
                return self.update_terminated_options(events, old_state)
            else:  # option has not terminated -> repetition would lead to an infinite loop until agent gets distracted
                print("Option", option, "did not terminate; option stack:", self.option_stack, "; no learning.")
                return []

    def update_v(self, option, goal_achieved):
        novelty = 1 / self.n[option[0], option[1]]
        RPE = goal_achieved * novelty - self.v[option[0], option[1]]
        self.v[option[0], option[1]] += self.alpha * RPE
        print("New v of option", option, ":", np.round(self.v, 2))

    def update_theta(self, option, goal_achieved, old_state):
        features = 1 - old_state[option[0]-1]
        thetas = self.o_theta[self.option_coord_to_index(option), option[1], :]
        RPE = goal_achieved - np.dot(thetas, features)
        self.o_theta[self.option_coord_to_index(option), option[1], :] += self.alpha * RPE * features
        print("New theta of option", option, ":",
              np.round(self.o_theta[self.option_coord_to_index(option), option[1], :], 2))

    def create_option(self, event):
        self.v[event[0], event[1]] = self.initial_value  # initialize new option's value
        self.option_stack.append(event)  # add new option to option stack (has already been reached this trial)
        print("Option", event, "with v[", event, "]", self.v[event[0], event[1]], "created!")

    def check_if_goal_achieved_distracted(self, option, events):
        goal_achieved = any([event == option for event in events])  # did the event occur that the option targets?
        distracted = self.distraction > np.random.rand()
        return [goal_achieved, distracted]

    def option_coord_to_index(self, coord):
        level, selected_a = coord
        if level == 0:
            n_options_below = - self.n_lights
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
