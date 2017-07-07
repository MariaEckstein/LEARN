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

class HierarchicalAgent(object):
    """
    This class encompasses all hierarchical agents.
    Hierarchical agents perceive higher-level lights and/or create options.
    """
    def __init__(self, alpha, epsilon, gamma, distraction, n_levels, n_lights, n_lights_tuple):

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
        self.initial_value = 0.5
        self.initial_theta = 0
        self.n = np.zeros([n_levels, n_lights])  # event counter
        self.v = self.initial_value * np.ones([n_levels, n_lights])  # feature weights (thetas) for actions and options
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
        [self.goal_achieved, self.distracted] = [False, False]
        self.history = []

    def take_action(self, old_state):
        print("FUNCTION take_action")
        if len(self.option_stack) == 0:  # not inside an option -> select option based on agent.v
            values = self.v.copy()
        else:  # inside an option -> select option based on in-option policy
            values = self.get_option_values(self.option_stack[-1], old_state)
        option = self.select_option(values)
        print("Selected option", option)
        if option[0] == 0:  # level == 0 means we selected a basic action -> return it right away
            return option
        else:  # level > 0 means we selected a higher-level option -> use its policy for option selection
            return self.take_action(old_state)

    def get_option_values(self, option, state):
        print("FUNCTION get_option_values")
        values = np.full([self.n_levels, self.n_lights], np.nan)  # initialize value array
        features = 1 - state[option[0]-1]  # features indicate which lights are OFF
        theta = self.o_theta[self.option_coord_to_index(option), :, :]
        print("Features:", features, "; theta:\n", theta)
        option_values = np.dot(theta, features)  # calculate values from thetas
        values[option[0]-1, :] = option_values
        print("option_values:", np.round(option_values, 2), "\nvalues:", np.round(values, 2), "\nfeatures:", features)
        return values

    def select_option(self, values):
        print("FUNCTION select_option")
        if np.random.rand() > self.epsilon:
            option = np.argwhere(values == np.nanmax(values))[0]
        else:
            available_options = np.argwhere(~ np.isnan(values))
            select = np.random.choice(range(len(available_options)))
            option = available_options[select]
        self.option_stack.append(option)  # put this option on top of the option stack
        self.history.append(option)  # for data analysis / debugging
        return option

    def learn(self, old_state, events, new_state):
        print("FUNCTION learn")
        # Handle events: update novelty & create options
        current_events = np.argwhere(events)
        for event in np.flipud(current_events):  # go through events from highest level to lowest
            self.n[event[0], event[1]] += 1  # novelty of event decreases
            if np.any(np.isnan(self.v[event[0], event[1]])):  # if option has not yet been discovered
                self.create_option(event)  # create option
        # Learn about terminated options: update v & theta
        self.update_terminated_options(old_state, current_events, new_state, [])

    def update_terminated_options(self, old_state, events, new_state, previous_option):
        print("FUNCTION update_terminated_options")
        print("option_stack:", self.option_stack)
        if len(self.option_stack) == 0:  # all options have terminated -> done
            return []
        else:  # there are options in the stack -> check if they have terminated, starting at the lowest-level one
            current_option = self.option_stack[-1]
            # goal_achieved = False
            [goal_achieved, distracted] = self.check_if_goal_achieved_distracted(current_option, events)  # SHOULD ALSO BE ABLE TO GET DISTRACTED IN HIGHER-LEVEL OPTIONS!
            if current_option[0] > 0:  # higher-level options update their in-option policy after each trial
                self.update_theta(current_option, goal_achieved, old_state, new_state, previous_option)
            print("current_option:", current_option, "; events:", events)
            print("goal_achieved:", goal_achieved, "; distracted:", distracted)
            if goal_achieved or distracted:  # current_option terminated -> update v and - for higher levels - thetas
                previous_option = self.option_stack.pop()
                self.update_v(current_option, goal_achieved)  # all options update their values
                if current_option[0] > 0:  # higher-level options also update their in-option policy
                    print("For current_option", current_option, "goal was achieved:", goal_achieved, "; got distracted:", distracted,
                          ". New theta_in_charge:\n", np.round(self.o_theta[self.option_coord_to_index(current_option), :, :], 2))
                return self.update_terminated_options(old_state, events, new_state, previous_option)
            else:  # current_option has not terminated -> repetition would lead to infinite loop
                print("current_option", current_option, "did not terminate; option stack:", self.option_stack, "; no learning.")
                return []

    def update_v(self, option, goal_achieved):
        print("FUNCTION update_v")
        novelty = 1 / self.n[option[0], option[1]]
        RPE = goal_achieved * novelty - self.v[option[0], option[1]]
        self.v[option[0], option[1]] += self.alpha * RPE
        print("New v (changed option", option, "using RPE", round(RPE, 2), "):\n", np.round(self.v, 2))

    def update_theta(self, current_option, goal_achieved, old_state, new_state, previous_option):
        print("FUNCTION update_theta")
        print("current_option:", current_option, "previous option:", previous_option)
        theta_previous_option = self.o_theta[self.option_coord_to_index(current_option), previous_option[1], :]
        features = 1 - old_state[current_option[0]-1]
        value_previous_option = np.dot(theta_previous_option, features)
        print("theta_previous_option:", theta_previous_option, "value_previous_option:", value_previous_option,
              "goal_achieved:", goal_achieved)
        value_next_option = 0  # TD!!! self.get_maxQ(current_option, new_state)
        RPE = goal_achieved + self.gamma * value_next_option - value_previous_option
        theta_previous_option += self.alpha * RPE * features
        print("New theta of current_option", current_option, ":", np.round(theta_previous_option, 2))
        print("Used RPE", RPE, "on these features:", features)

    def get_maxQ(self, current_option, new_state):
        print("FUNCTION get_maxQ")
        values = self.get_option_values(current_option, new_state)  # get in-option values of current option
        maxQ = np.nanmax(values)
        print("maxQ:", maxQ)
        return maxQ

    def create_option(self, event):
        print("FUNCTION create_option")
        self.v[event[0], event[1]] = self.initial_value  # initialize new option's value
        self.option_stack.insert(-1, event)  # add new option at position before basic action (has already been reached)
        print("Created option", event)

    def check_if_goal_achieved_distracted(self, option, events):
        print("FUNCTION check_of_goal_achieved_distracted")
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

