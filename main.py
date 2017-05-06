
import math
from flat_agents import RewardAgent, NoveltyAgent, NoveltyRewardAgent
from environment import Environment

n_lights = 18   # number of level-0 lights
alpha = 0.75
epsilon = 0.1
n_trials = 200
n_lights_tuple = 3  # number of lights per level-0 tuple
n_levels = math.ceil(n_lights ** (1/n_lights_tuple))


agent = NoveltyRewardAgent(alpha, epsilon, n_lights)
environment = Environment(n_levels, n_lights, n_lights_tuple)

for _ in range(n_trials):
    old_state = environment.state.copy()  # ohne copy waeren old_state und new_state nur andere namen fuer environment.state
    print(agent.v)
    action = agent.take_action(old_state)
    print(action)
    environment.respond(action)
    new_state = environment.state
    agent.update_values(old_state, action, new_state)
    print(new_state)