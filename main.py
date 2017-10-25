# Define variables
data_dir = 'C:/Users/maria/MEGAsync/Berkeley/LEARN/data/'
n_trials = 20
n_agents = 1
n_envs = 1
env_stuff = {'option_length': 2,
             'n_options_per_level': [3, 3, 3, 3, 3, 3]}
agent_stuff = {'hier_level': len(env_stuff['n_options_per_level']),  # flat (0), hierarchical (len(env_stuff['n_options_per_level'])), in-between?
               'learning_signal': 'novelty',  # novelty or reward
               'alpha': 0.3,  # learning rate
               'e_lambda': 0.5,  # how much of the elig. trace is retained from trial to trial?
               'n_lambda': 0.3,  # how fast does novelty decay?
               'gamma': 0.7,  # how much does the agent care about the future?
               'epsilon': 0.2,  # what percentage of actions is selected randomly?
               'distraction': 0.2}  # probability of quitting an option at each step


# Define agent play as a function
def let_agent_play(n_trials, n_agents, env, agent_stuff, data_dir):

    from hierarchical_agents import Agent
    from history import History

    # Let agents play the game
    ag = 0
    while ag <= n_agents:
        agent_stuff['id'] = ag
        agent = Agent(agent_stuff, env)
        hist = History(env, agent)
        try:
            for trial in range(n_trials):
                state_before = env.state.copy()
                action = agent.take_action(state_before, trial, hist, env)
                events = env.make_events(action, hist, trial)
                state_after = env.state.copy()
                agent.learn(hist, env, events, trial, state_before, state_after)
            # Save agent data
            hist.save_all(agent, env, data_dir)
            print('Agent', ag)
            ag += 1
        except:
            pass


# Execute the function: let different agents play in different environments!
from environment import Environment
for env_id in range(n_envs):

    print('Environment', env_id)
    env_stuff['id'] = env_id
    env = Environment(env_stuff, n_trials)

    print('Hierarchical-novelty agent')
    agent_stuff['hier_level'] = len(env_stuff['n_options_per_level'])
    agent_stuff['learning_signal'] = 'novelty'
    let_agent_play(n_trials, n_agents, env, agent_stuff, data_dir)

    print('Hierarchical-reward agent')
    agent_stuff['hier_level'] = len(env_stuff['n_options_per_level'])
    agent_stuff['learning_signal'] = 'reward'
    let_agent_play(n_trials, n_agents, env, agent_stuff, data_dir)

    print('Flat-novelty agent')
    agent_stuff['hier_level'] = 0
    agent_stuff['learning_signal'] = 'novelty'
    let_agent_play(n_trials, n_agents, env, agent_stuff, data_dir)

    print('Flat-reward agent')
    agent_stuff['hier_level'] = 0
    agent_stuff['learning_signal'] = 'reward'
    let_agent_play(n_trials, n_agents, env, agent_stuff, data_dir)
