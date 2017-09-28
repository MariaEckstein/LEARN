def let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir):

    from hierarchical_agents import Agent
    from environment import Environment
    from history import History
    import numpy as np
    import pandas as pd
    import os

    # Let agent play the game and record data
    for ag in range(n_agents):
        env = Environment(env_stuff, n_trials)
        agent = Agent(agent_stuff, env)
        hist = History(env, agent)
        for trial in range(n_trials):
            old_state = env.state.copy()
            hist.state[trial, :, :] = old_state
            action = agent.take_action(old_state, hist)
            agent.action_history[trial, action[1]] = 1
            env.switch_lights(action)
            events = env.make_events(action)
            hist.event[trial, :, :] = events.copy()
            agent.learn(old_state, events, hist)
            agent.trial += 1

    # Save data
    data_path = hist.get_data_path(agent, env, data_dir)
    if not os.path.isdir(data_path):
        os.makedirs(data_path)

    hist.save_events(env, data_path)
    hist.save_states(env, data_path)
    hist.save_v(env, data_path)
    hist.save_theta(env, data_path)

    # Save option history to csv
    colnames = [str(i) for i in range(env.n_lights)]
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
agent_stuff = {'hier_level': 0,
               'learning_signal': 'reward',
               'alpha': 0.2,
               'lambd': 0.5,
               'epsilon': 0.2,
               'distraction': 0}
data_dir = 'C:/Users/maria/MEGAsync/Berkeley/LEARN/data/'

env_stuff = {'n_lights': 4,
             'n_lights_tuple': 2}
let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir)
#
# env_stuff = {'n_lights': 16,
#              'n_lights_tuple': 2}
# let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir)

# env_stuff = {'n_lights': 27,
#              'n_lights_tuple': 3}
# let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir)
