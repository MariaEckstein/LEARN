from collections import defaultdict
n_blue_lights = 9
n_lights = n_blue_lights + n_blue_lights // 3 + n_blue_lights // 9
alpha = 0.5


class Environment(object):
    def __init__(self):
        self.state = [False] * n_lights # creates array of len n_lights

    def perform(self, action):
        lightNumber, switchTo = action # action in tuple-form: action = (lightNumber, switchTo)
        self.state[lightNumber] = switchTo
        self.do_events(lightNumber)

    # do_events ist noch nicht ausgereift, das schaltet die triple lichter aus und so
    def do_events(self, lightNumber):
        first_in_triple = lightNumber - (lightNumber % 3)
        triple = range(first_in_triple, first_in_triple + 3)
        if all(self.state[i] for i in triple):
            yellow_light = n_blue_lights + lightNumber // 3
            self.state[yellow_light] = True
            for i in triple:
                self.state[i] = False
        yellow_triple = range(n_blue_lights + 1, n_blue_lights + 3)
        if all(self.state[i] for i in yellow_triple):
            self.state[-1] = True
            for i in yellow_triple:
                self.state[i] = False


class Agent(object):
    def __init__(self):
        default = 0
        self.v = defaultdict(lambda: default) # v stores values for action tuples: v[(1,False)] = value of switching light 1 off
        # defaultdict throws no error when a missing item is requested, instead it returns 0 (all v have 0 as default)


class RewardAgent(Agent):
    def take_action(self, state):
        possible_actions = [(i, True) if not state[i] else (i, False) for i in range(n_blue_lights)]
        return max(possible_actions, key=lambda a: self.v[a]) # simply return action with highest v

    def update(self, old_state, action, new_state):
        reward = sum(new_state) - sum(old_state)
        self.v[action] += alpha * (reward - self.v[action])


agent = RewardAgent()
environment = Environment()

for _ in range(30):
    old_state = environment.state.copy() # ohne copy wären old_state und new_state nur andere namen für environment.state
    action = agent.take_action(old_state)
    environment.perform(action)
    new_state = environment.state
    agent.update(old_state, action, new_state)
    print(new_state)