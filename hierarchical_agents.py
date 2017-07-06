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
        self.theta = self.initial_value * np.ones([n_levels, n_lights, n_lights])  # feature weights (thetas) for actions and options
        self.theta[1:] = np.nan  # undefined for options
        self.o_theta = np.full([self.n_options - n_lights + 1, n_lights, n_lights], np.nan)  # in-option values
        row_ot = 0
        for level in range(n_levels):
            n_options_level = n_lights // (2 * n_lights_tuple ** level)
            for option in range(n_options_level):
                n_lights_level = n_lights // n_lights_tuple ** level  # number of lights at level below option
                self.o_theta[row_ot, range(n_lights_level), :] = self.initial_value
                row_ot += 1

        # Agent's memory about his plans and past actions
        self.option_stack = []  # contains the o_theta rows of each option that is currently being executed
        self.level = 0  # at which level are we in the option?
        self.theta_in_charge = self.theta.copy()
        self.options = [list() for _ in range(n_levels)]  # options taken; one list per level
        [self.goal_achieved, self.distracted] = [False, False]

    def take_action(self, old_state):
        # self.create_options(state, events)  # NOW IN LEARN! if an event happened, create the corresponding option
        if len(self.option_stack) == 0:  # we're not inside an option -> choose among all the possibilities
            # self.update_theta(state)  # NOW IN LEARN! update agent.theta according to last action and outcome
            # self.theta_in_charge = self.theta.copy()  # NOW IN LEARN!
            option = self.select_option(old_state)
            self.level = option[0]
            print("Had free choice and selected option", option)
        else:  # we're inside an option -> follow option policy
            print("Inside option", self.option_stack)
            option = [self.level, self.select_option(old_state)[1]]
            # option = self.learn_option(state, events)  # update o_theta; check if option terminates, if yes update theta
        print("Option:", option)
        action = self.get_basic_action(option, old_state)  # trickle down hierarchy until we reach basic level
        self.n[action[0], action[1]] += 1  # a basic action always reaches its subgoal; events always update n
        print("Action:", action)
        self.options[self.level].append(action[1])
        print("Agent.options:", self.options)
        return action

    def learn(self, old_state, events):
        print("Events:", np.argwhere(events))
        self.create_options(old_state, events)
        if len(self.option_stack) == 0:  # we're not inside an option -> choose among all the possibilities
            print("Learning outside options (len(self.option_stack) == 0).")
            self.update_theta(old_state)  # update agent.theta according to last action and outcome
            self.theta_in_charge = self.theta.copy()
        else:
            self.learn_option(old_state, events)  # if option ends, update_ot & update_theta & update theta_in_charge

    def create_options(self, old_state, events):
        current_events = np.argwhere(events)
        for event in current_events:
            self.n[event[0], event[1]] += 1  # record the event
            if np.any(np.isnan(self.theta[event[0], event[1], :])):  # if option has not yet been discovered
                print("state[", (event[0]-1), "]:", old_state[(event[0] - 1)])  # SHOULD WORK NOW! DOESN'T WORK because the executed actions are off in state
                # Learn the value of the option
                self.theta[event[0], event[1], :] = self.initial_value  # initialize it
                RPE = 1 - self.theta[event[0], event[1], :]
                features = 1 - old_state[(event[0] - 1)]
                self.theta[event[0], event[1], :] += self.alpha * RPE * features  # and learn
                print("Option", self.option_coord_to_index(event), event, "with theta[", event[0], event[1], ", :]",
                      self.theta[event[0], event[1], :], "created!")
                # Learn the in-option policy
                row = self.option_coord_to_index(event)
                col = self.options[self.level][-1]
                RPE = 1 - self.o_theta[row, event[1], :]
                self.o_theta[row, col, :] += self.alpha * RPE * features
                print("Option", self.option_coord_to_index(event), event, "has o_theta[", row, col, ", :]",
                      self.o_theta[row, col, :])

    def update_ot(self, goal_achieved, old_state):
        previous_option = [self.level, self.options[self.level][-1]]
        print("Previous ot of option", self.option_stack[-1], ":", np.round(self.o_theta[self.option_stack[-1], :], 2))
        index = [self.option_stack[-1], previous_option[1]]
        self.rescorla_wagner(old_state, index, goal_achieved, self.o_theta)
        print("New ot of option", self.option_stack[-1], ":", np.round(self.o_theta[self.option_stack[-1], :], 2))

    def update_theta(self, old_state):
        print("Previous theta:", np.round(self.theta, 2))
        print("N:", self.n)
        if len(self.options[self.level]) > 0:  # we have previously executed an action/option at this level
            if len(self.option_stack) == 0:  # we're not inside an option -> update basic actions
                index = [self.level, self.options[self.level][-1]]
                print("Updating action", index)
            else:  # we are inside an option -> update options
                index = [self.level+1, self.options[self.level+1][-1]]
                print("Updating option", index)
            novelty = 1 / self.n[index[0], index[1]]  # SHOULD BE TOTAL NOVELTY OF TRIAL, I.E., LEN(EVENTS)?
            self.rescorla_wagner(old_state, index, novelty, self.theta)
        print("New theta:", np.round(self.theta, 2))

    def rescorla_wagner(self, old_state, index, reward, theta):
        RPE = reward - theta[index[0], index[1], :]
        features = 1 - old_state[self.level]  # features indicate which lights are OFF; state indicates which are ON
        theta[index[0], index[1], :] += self.alpha * RPE * features  # SIDE EFFECTS!

    def learn_option(self, old_state, events):
        [self.goal_achieved, self.distracted] = self.check_if_goal_achieved_distracted(events)
        if self.goal_achieved or self.distracted:
            self.update_ot(self.goal_achieved, old_state)
            self.update_theta(old_state)
            self.option_stack.pop()  # check off current option
            print("Goal achieved:", self.goal_achieved, "; got distracted:", self.distracted, "; will now do", self.option_stack)
            if len(self.option_stack) == 0:
                self.theta_in_charge = self.theta.copy()
            else:
                self.theta_in_charge = self.o_theta[self.option_stack[-1], :]
                self.level += 1
                return self.learn_option(old_state, events)
        else:
            print("Still inside option(s)", self.option_stack, ", so no learning.")

    def check_if_goal_achieved_distracted(self, events):
        current_events = np.argwhere(events)
        current_events_i = np.array([self.option_coord_to_index(current_events[i]) for i in range(len(current_events))])
        goal_achieved = np.any(current_events_i == self.option_stack[-1])
        distracted = self.distraction > np.random.rand()
        return [goal_achieved, distracted]

    def select_option(self, old_state):
        features = 1 - old_state[self.level]  # features indicate which lights are OFF
        values = np.dot(self.theta_in_charge, features)  # WORKING? calculate values from thetas
        if len(values.shape) == 1:
            nans = np.full([1, len(values)], np.nan)
            values = np.vstack([values, nans])
        if np.random.rand() > self.epsilon:
            option = np.argwhere(values == np.nanmax(values))[0]
        else:
            available_options = np.argwhere(~ np.isnan(values))
            select = np.random.choice(range(len(available_options)))
            option = available_options[select]
        return option

    def get_basic_action(self, option, old_state):
        if self.level == 0:
            return option
        else:
            print("Trickling down levels... now in option", self.option_stack, option)
            self.options[self.level].append(option[1])
            option_i = self.option_coord_to_index(option)
            self.option_stack.append(option_i)
            self.level -= 1
            self.theta_in_charge = self.o_theta[self.option_stack[-1], :]  # get the values of the new option
            print("Option", self.option_stack[-1], "theta_in_charge:", np.round(self.theta_in_charge, 2))
            option = [self.level, self.select_option(old_state)[1]]
            print(self.level)
            return self.get_basic_action(option, old_state)

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
