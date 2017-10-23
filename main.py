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
            state_before = env.state.copy()
            action = agent.take_action(state_before, trial, hist, env)
            events = env.make_events(action, hist, trial)
            state_after = env.state.copy()
            agent.learn(hist, env, events, trial, state_before, state_after)

    agent.theta.get()

    # Save data
    hist.save_all(data_dir, env, agent)


# Execute the function: let the agent play!
n_trials = 200
n_agents = 1
env_stuff = {'option_length': 2,
             'n_options_per_level': [3, 3, 3, 3, 3, 3]}
agent_stuff = {'hier_level': len(env_stuff['n_options_per_level']),  # flat (0), hierarchical (len(env_stuff['n_options_per_level'])), in-between?
               'learning_signal': 'novelty',  # novelty or reward
               'alpha': 0.3,  # learning rate
               'e_lambda': 0.5,  # how much of the elig. trace is retained from trial to trial?
               'n_lambda': 0.3,  # how fast does novelty decay?
               'gamma': 0.9,  # how much does the agent care about the future?
               'epsilon': 0.2,  # what percentage of actions is selected randomly?
               'distraction': 0.1}  # probability of quitting an option at each step
data_dir = 'C:/Users/maria/MEGAsync/Berkeley/LEARN/data/'

let_agent_play(n_trials, n_agents, agent_stuff, env_stuff, data_dir)
