def let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir):

    from hierarchical_agents import Agent
    from environment import Environment
    from history import History

    # Let agent play the game
    for ag in range(n_agents):
        env = Environment(env_stuff, n_trials)
        agent = Agent(agent_stuff, env)
        hist = History(env, agent)
        for trial in range(n_trials):
            old_state = env.state.copy()
            action = agent.take_action(old_state, hist, env)
            events = env.make_events(action, hist, trial)
            agent.learn(events, hist, env)
            agent.trial += 1

    # Save data
    hist.save_all(data_dir, env, agent)


# Execute the function: let the agent play!
n_trials = 200
n_agents = 1
agent_stuff = {'hier_level': 0,  # flat (0), hierarchical (len(n_options_per_level)), in-between?
               'learning_signal': 'novelty',
               'alpha': 0.3,  # learning rate
               'e_lambda': 0.5,  # how fast does the elig. trace evaporate?
               'n_lambda': 0.2,  # how fast does novelty decay?
               'gamma': 0.8,  # how much does the agent care about the future?
               'epsilon': 0.1,  # what percentage of actions is selected randomly?
               'distraction': 0}  # probability of quitting an option at each step
data_dir = 'C:/Users/maria/MEGAsync/Berkeley/LEARN/data/'

env_stuff = {'option_length': 2,
             'n_options_per_level': [4, 4, 4]}

let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir)
