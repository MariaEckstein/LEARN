def let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir):

    from hierarchical_agents import Agent
    from environment import Environment
    from history import History
    import pandas as pd
    import numpy as np
    import os

    # Let agent play the game and record data
    for ag in range(n_agents):
        env = Environment(env_stuff, n_trials)
        agent = Agent(agent_stuff, env)
        hist = History(env, agent)
        for trial in range(n_trials):
            old_state = env.state.copy()
            action = agent.take_action(old_state, hist, env)
            events = env.make_events(action, hist, trial)
            hist.state[trial, :, :] = env.state.copy()
            agent.learn(events, hist, env)
            hist.e[trial] = agent.theta.e.copy()
            agent.trial += 1

    # Save data
    data_path = hist.get_data_path(agent, env, data_dir)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    hist.save_e(env, data_path)
    hist.save_events(env, data_path)
    hist.save_states(env, data_path)
    hist.save_v(env, data_path)
    hist.save_theta(env, data_path)

    # Save option history to csv
    colnames = [str(i) for i in range(env.n_basic_actions)]
    colnames = colnames + ['trial', 'step']
    option_history = pd.DataFrame(columns=colnames)
    for row in range(agent.option_history.shape[0]):
        step_history = pd.DataFrame(agent.option_history[row, :, :], columns=colnames)
        step_history['level'] = range(env.n_levels)
        option_history = pd.concat([option_history, step_history])
    option_history.head()
    option_history_long = pd.melt(option_history, id_vars=["trial", "step", "level"], var_name="action")
    option_history_long['action'] = pd.to_numeric(option_history_long['action'])
    option_history_long.to_csv(data_path + "/option_history_long.csv")


n_trials = 500
n_agents = 1
agent_stuff = {'hier_level': 4,
               'learning_signal': 'novelty',
               'alpha': 0.3,  # learning rate
               'e_lambda': 0.5,  # how fast does the elig. trace evaporate?
               'n_lambda': 0.2,  # how fast does novelty decay?
               'gamma': 0.8,  # how much does the agent care about the future?
               'epsilon': 0.1,  # what percentage of actions is selected randomly?
               'distraction': 0}
data_dir = 'C:/Users/maria/MEGAsync/Berkeley/LEARN/data/'

env_stuff = {'option_length': 2,
             'n_options_per_level': [4, 4, 4, 4]}

let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir)
