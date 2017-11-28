from environment import Environment
from hierarchical_agents import Agent
from history import History
import numpy as np


def let_agent_play(agent, env, trial, external_rewards):
    state_before = env.state.copy()
    action = agent.take_action(state_before, trial, hist, env)
    events = env.make_events(action, hist, trial)
    rewards = env.give_rewards(external_rewards)
    state_after = env.state.copy()
    agent.learn(hist, env, events, rewards, trial, state_before, state_after)

def the_whole_thing(agent, env, hist):
    for trial in range(0, n_trials['play']):
        let_agent_play(agent, env, trial, external_rewards=False)
    for trial in range(n_trials['play'], n_trials['play'] + n_trials['reward']):
        let_agent_play(agent, env, trial, external_rewards=True)
    print('Saving agent', ag, '...')
    hist.save_all(agent, env, data_dir)


# Define specifics
data_dir = 'C:/Users/maria/MEGAsync/Berkeley/LEARN/data/2017_11_28'
n_trials = {'play': 200,
            'reward': 200}
n_agents = 50
n_envs = 50
option_length = 2
n_options_per_level = [5, 5, 5, 5, 5]
reward = {'events': [np.random.choice(range(n), 2, replace=False) for n in n_options_per_level],
          'value': 0.5}
parameters = {'alpha': 0.3,  # learning rate
              'n_lambda': 0.3,  # how fast does novelty decay?
              'gamma': 0.9,  # how much does the agent care about the future?
              'epsilon': 0.2,  # what percentage of actions is selected randomly?
              'distraction': 0.1}  # probability of quitting an option at each step

# Let different agents play in different environments
for env_id in range(n_envs):
    print('Environment', env_id)
    env = Environment(option_length, n_options_per_level, reward, n_trials, env_id)

    print('Hierarchical-novelty agents')
    for ag in range(n_agents):
        agent = Agent(parameters, 'novelty', len(n_options_per_level), ag, env)
        hist = History(env, agent)
        the_whole_thing(agent, env, hist)

    print('Hierarchical-reward agents')
    for ag in range(n_agents):
        agent = Agent(parameters, 'reward', len(n_options_per_level), ag, env)
        hist = History(env, agent)
        the_whole_thing(agent, env, hist)

    print('Flat-novelty agents')
    for ag in range(n_agents):
        agent = Agent(parameters, 'novelty', 0, ag, env)
        hist = History(env, agent)
        the_whole_thing(agent, env, hist)

    print('Flat-reward agents')
    for ag in range(n_agents):
        agent = Agent(parameters, 'reward', 0, ag, env)
        hist = History(env, agent)
        the_whole_thing(agent, env, hist)
