from collections import defaultdict
import numpy as np
import math
n_blue_lights = 9
n_lights_lev0_tuple = 3
n_colors = math.ceil(n_blue_lights ** (1/n_lights_lev0_tuple))
# n_lights = n_blue_lights + n_blue_lights // 3 + n_blue_lights // 9
alpha = 0.5


class Environment(object):
    def __init__(self):
        self.state = np.zeros((n_colors, n_blue_lights))

    def respond(self, action):
        lightNumber, switchTo = action  # action in tuple-form: action = (lightNumber, switchTo)
        self.state[0, lightNumber] = switchTo
        self.do_events(lightNumber)

    def do_events(self, lightNumber):
        n_levels = self.state.shape[0] - 1
        for level in range(0, n_levels):  # "level" means color: blue, yellow, green, etc.
            n_lights_tuple = n_lights_lev0_tuple * 3 ** level
            first_in_tuple = lightNumber - (lightNumber % n_lights_tuple)
            color_tuple = range(first_in_tuple, first_in_tuple + n_lights_tuple)
            if all(self.state[level, color_tuple]):  # if all lights in a tuple of this level are on
                self.state[level + 1, color_tuple] = True  # turn next-level light on; take multiple columns
                self.state[level, color_tuple] = 0  # turn lower-level lights off


class Agent(object):
    def __init__(self):
        default = 0
        self.v = defaultdict(lambda: default)  # v stores values for action tuples: v[(1,False)] = value of switching light 1 off
        # defaultdict throws no error when a missing item is requested, instead it returns 0 (all v have 0 as default)


class RewardAgent(Agent):
    def take_action(self, state):
        possible_actions = [(i, False) if state[0, i] else (i, True) for i in range(n_blue_lights)]
        return max(possible_actions, key=lambda a: self.v[a]) # simply return action with highest v

    def update_values(self, old_state, action, new_state):
        reward = sum(new_state[0,:]) - sum(old_state[0,:])
        self.v[action] += alpha * (reward - self.v[action])


agent = RewardAgent()
environment = Environment()

for _ in range(30):
    old_state = environment.state.copy() # ohne copy waeren old_state und new_state nur andere namen fuer environment.state
    action = agent.take_action(old_state)
    print(action)
    environment.respond(action)
    new_state = environment.state
    agent.update_values(old_state, action, new_state)
    print(new_state)
    