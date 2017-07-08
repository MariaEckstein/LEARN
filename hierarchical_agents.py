import numpy as np

# Update:
# - took out OFF action
# - options are working
# - did features instead of states

# Curiosity & Hierarchy / structure: How do we learn to understand the world? -> We stop seeing basic actions and start
# seeing options. We associate elements of the world into bigger and bigger chunks. We understand the world in terms
# of what we can do with it. A baby does not see the computer screen, mouse, and key board, but black and shiny boxes
# of different sizes. Humans explore the world in a smart way, trying to understand it, trying to reduce unexpected
# novelty by seeking previously novel events, learning their underlying mechanisms. This is a very general
# description of human learning and exploration that applies to many situations: motor learning, language learning,
# concept learning. Previous research has shown that humans infer hierarhcical structure (Collins,
# Botvinick, Badre, Frank) and love novelty (Gershman & Niv, 2015).

# Conferences: cognitive science conference / artificial general intelligence; cognitive systems

# Content:
# don't compare algo to people directly, make the point that it's the same kind that drives both
# my algo should show a systematic search through the space, do people do a similar thing?
# compare to an agent that is not driven by novelty -> by what else could it be driven? "reward", i.e., difference in the number of lights?
# compare to an agent without options -> gets bored, has unstructured behavior
# trajectory over time -> behavior is unstructured at first, becomes more and more structured
# compare worlds of different sizes, different numbers of levels, plot learning (difference between current and optimal values) over time in each
# play with different numbers of levels -> how many levels do humans have?

# TDs:
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
    def __init__(self, alpha, epsilon, gamma, distraction, n_levels, n_lights, n_lights_tuple, display_mode):

        self.display_mode = display_mode
        # Agent's RL features
        self.alpha = alpha  # learning rate
        self.epsilon = epsilon  # greediness
        self.gamma = gamma  # future discounting
        self.distraction = distraction  # propensity to quit unfinished options

        # Characteristics of the environment
        self.n_levels = n_levels
        self.n_lights = n_lights
        self.n_lights_tuple = n_lights_tuple
        self.n_options = np.sum([n_lights // n_lights_tuple ** i for i in range(n_levels)])

        # Agent's thoughts about its environment (values, novelty)
        self.initial_value = 1 / n_lights_tuple / 2
        self.initial_theta = 1 / n_lights / 2
        self.n = np.zeros([n_levels, n_lights])  # event counter
        self.v = self.initial_value * np.ones([n_levels, n_lights])  # values of actions and options
        self.v[1:] = np.nan  # undefined for options
        self.o_theta = np.full([self.n_options - n_lights + 1, n_lights, n_lights], np.nan)  # in-option features
        row_ot = 0
        for level in range(n_levels):
            n_options_level = n_lights // (2 * n_lights_tuple ** level)
            for option in range(n_options_level):
                n_lights_level = n_lights // n_lights_tuple ** level  # number of lights at level below option
                self.o_theta[row_ot, range(n_lights_level), :] = self.initial_theta
                row_ot += 1

        # Agent's memory about his plans and past actions
        self.option_stack = []  # stack of the option(s) that are currently guiding behavior
        self.history = []

    def take_action(self, old_state):
        if self.display_mode == "extra-verbose":
            print("FUNCTION take_action")

        if len(self.option_stack) == 0:  # not inside an option -> select option based on agent.v
            values = self.v.copy()
        else:  # inside an option -> select option based on in-option policy
            values = self.get_option_values(self.option_stack[-1], old_state)
        option = self.select_option(values)
        if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
            print("Selected option", option)
        if option[0] == 0:  # level == 0 means we selected a basic action -> return it right away
            return option
        else:  # level > 0 means we selected a higher-level option -> use its policy for option selection
            return self.take_action(old_state)

    def get_option_values(self, option, state):
        if self.display_mode == "extra-verbose":
            print("FUNCTION get_option_values")

        features = 1 - state[option[0]-1]  # features indicate which lights are OFF
        theta = self.o_theta[self.option_coord_to_index(option), :, :]
        option_values = np.dot(theta, features)  # calculate values from thetas
        values = np.full([self.n_levels, self.n_lights], np.nan)  # initialize value array
        values[option[0]-1, :] = option_values
        if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
            print("features:", features, "\ntheta:\n", np.round(theta, 2), "\nvalues:\n", np.round(values, 2))
        return values

    def select_option(self, values):
        if self.display_mode == "extra-verbose":
            print("FUNCTION select_option")

        if np.random.rand() > self.epsilon:
            selected_options = np.argwhere(values == np.nanmax(values))  # all options with the highest value
        else:
            selected_options = np.argwhere(~ np.isnan(values))  # all options that are not nan
        select = np.random.choice(range(len(selected_options)))  # randomly select the index of one of the options
        option = selected_options[select]  # pick that option
        self.option_stack.append(option)  # put this option on top of the option stack
        self.history.append(option)  # for data analysis / debugging
        return option

    def learn(self, old_state, events, new_state):
        if self.display_mode == "extra-verbose":
            print("FUNCTION learn")

        current_events = np.argwhere(events)
        self.learn_from_events(old_state, current_events, new_state)  # count events, create options, update thetas
        self.update_options(old_state, current_events, new_state, [])  # update v of term. options, theta of non-term.

    def learn_from_events(self, old_state, current_events, new_state):
        if self.display_mode == "extra-verbose":
            print("FUNCTION learn_from_events")

        previous_option = []
        for event in current_events:
            self.n[event[0], event[1]] += 1  # update event count (novelty decreases)
            if np.any(np.isnan(self.v[event[0], event[1]])):  # if option has not yet been discovered
                self.v[event[0], event[1]] = self.initial_value  # create option
                self.update_v(event, 1)  # update option value right away
                if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
                    print("Created option", event)
            if event[0] > 0:  # if it's a higher-level event
                self.update_theta(event, 1, old_state, new_state, previous_option)  # update in-option policy
            previous_option = event.copy()  # it's not cheating...

    def update_options(self, old_state, events, new_state, previous_option):
        if self.display_mode == "extra-verbose":
            print("FUNCTION update_options")

        for current_option in np.flipud(self.option_stack):  # go through options, starting at lowest-level one
            [goal_achieved, distracted] = self.check_if_goal_achieved_distracted(current_option, events)
            if current_option[0] > 0 and not goal_achieved:  # higher-level options update in-option policy each trial
                self.update_theta(current_option, 0, old_state, new_state, previous_option)
            if goal_achieved or distracted:  # current_option terminated -> update v
                previous_option = self.option_stack.pop()
                self.update_v(current_option, goal_achieved)  # update values of options after termination
                if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
                    print("Updating v because goal achieved:", goal_achieved, "; distracted:", distracted)
            else:  # current_option has not terminated -> repetition would lead to infinite loop
                if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
                    print("current_option", current_option, "did not terminate; option stack:", self.option_stack)
                break

    def update_v(self, option, goal_achieved):
        if self.display_mode == "extra-verbose":
            print("FUNCTION update_v")

        novelty = 1 / self.n[option[0], option[1]]
        RPE = goal_achieved * novelty - self.v[option[0], option[1]]
        self.v[option[0], option[1]] += self.alpha * RPE
        if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
            print("v after updating option", option, "using RPE", round(RPE, 2), ":\n", np.round(self.v, 2))

    def update_theta(self, current_option, goal_achieved, old_state, new_state, previous_option):
        if self.display_mode == "extra-verbose":
            print("FUNCTION update_theta")

        theta_previous_option = self.o_theta[self.option_coord_to_index(current_option), previous_option[1], :]
        features = 1 - old_state[current_option[0]-1]
        value_previous_option = np.dot(theta_previous_option, features)
        if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
            print("Option stack:", self.option_stack,
                  "\nthetas of option", previous_option, "inside option", current_option, ":",
                  np.round(theta_previous_option, 2),
                  "\nfeatures:", features,
                  "\nvalue of option", previous_option, "inside option", current_option, ":",
                  np.round(value_previous_option, 2),
                  "\nachieved goal of option", current_option, "by taking option", previous_option, ":", goal_achieved)
        value_next_option = 0  # TD!!! self.get_maxQ(current_option, new_state)
        RPE = goal_achieved + self.gamma * value_next_option - value_previous_option
        theta_previous_option += self.alpha * RPE / sum(features) * features
        if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
            print("New theta of option", previous_option, "inside option", current_option, ":",
                  np.round(theta_previous_option, 2),
                  "\nall thetas of option", current_option, ":\n",
                  np.round(self.o_theta[self.option_coord_to_index(current_option), :, :], 2))

    def check_if_goal_achieved_distracted(self, option, events):
        goal_achieved = any([np.all(option == event) for event in events])  # did event occur that option targets?
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

    # def get_maxQ(self, current_option, new_state):
    #     if self.display_mode == "extra-verbose":
    #         print("FUNCTION get_maxQ")
    #     values = self.get_option_values(current_option, new_state)  # get in-option values of current option
    #     maxQ = np.nanmax(values)
    #     if any(self.display_mode == np.array(["verbose", "extra-verbose"])):
    #         print("maxQ:", maxQ)
    #     return maxQ


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

