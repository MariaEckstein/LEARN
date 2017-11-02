import numpy as np
import pandas as pd
import os


class History(object):
    def __init__(self, env, agent):
        # Info for saving dataframes
        self.data_path = ''
        self.env_id = env.id
        self.agent_id = agent.id
        # Dataframes to be saved
        self.state = np.zeros([env.n_trials, env.n_levels, env.n_basic_actions])
        self.event = np.zeros(self.state.shape)
        self.event_s = np.full([env.n_trials, env.n_levels], np.nan)  # list of past events at each level
        self.action_s = np.full(self.event_s.shape, np.nan)  # list of past actions at each level
        self.v = np.zeros([env.n_trials, env.n_levels, env.n_basic_actions])
        self.n = np.zeros(self.v.shape)
        theta_shape = list(agent.theta.theta.shape)
        theta_shape[-1] += 2
        theta_shape.insert(0, env.n_levels * env.n_trials)
        self.theta = np.zeros(theta_shape)
        self.theta_row = 0
        self.option = np.zeros([env.n_trials * env.n_levels, env.n_levels, env.n_basic_actions + 2])
        self.option_row = 0
        self.step = 0
        self.e = np.zeros(np.append(env.n_trials, agent.theta.theta.shape))  # elig. trace

    def save_all(self, agent, env, data_dir):
        self.get_data_path(agent, env, data_dir)
        if not os.path.isdir(self.data_path):
            os.makedirs(self.data_path)
        self.save_rules(env)
        self.save_rest(env, 'state')
        self.save_rest(env, 'event')
        self.save_rest(env, 'v')
        self.save_rest(env, 'n')
        # self.save_theta(env)
        # self.save_options(env)
        # self.save_e(env)

    def get_data_path(self, agent, env, data_dir):
        hier = 'hierarchical' if agent.hier_level > 0 else 'flat'
        folder_structure = "/bugfix_" +\
                           "/" + hier +\
                           "/" + agent.learning_signal + "/"
        self.data_path = data_dir + folder_structure

    def save_rules(self, env):
        colnames = ['action' + str(i) for i in range(env.option_length)]
        rules = pd.DataFrame(columns=colnames)
        op = 0
        for level in range(1, env.n_levels):
            trial_rules = pd.DataFrame(env.rules[level, :, :], columns=colnames)
            trial_rules['option'] = range(op, op + env.n_options_per_level[level])
            op += env.n_options_per_level[level]
            trial_rules['level'] = level
            rules = pd.concat([rules, trial_rules])
        rules.to_csv(self.data_path + "/rules_e" + str(self.env_id) + ".csv")

    def save_theta(self, env):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        colnames = colnames + ['trial', 'updated_option']
        theta_history = pd.DataFrame(columns=colnames)
        for row in range(self.theta.shape[0]):
            for option in range(self.theta.shape[1]):
                option_theta_history = pd.DataFrame(self.theta[row, option, :, :], columns=colnames)
                option_theta_history['option'] = option
                option_theta_history['action'] = range(env.n_basic_actions)
                theta_history = pd.concat([theta_history, option_theta_history])
        theta_history = theta_history[theta_history['1'] != 0]
        theta_history_long = pd.melt(theta_history, id_vars=["trial", "option", "action", "updated_option"],
                                     var_name="feature")
        theta_history_long['feature'] = pd.to_numeric(theta_history_long['feature'])
        theta_history_long.to_csv(self.data_path + "/theta_history_long_e" + str(self.env_id) + "_a" + str(self.agent_id) + ".csv")

    def save_options(self, env):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        colnames = colnames + ['trial', 'step']
        option_history = pd.DataFrame(columns=colnames)
        for row in range(self.option.shape[0]):
            step_history = pd.DataFrame(self.option[row, :, :], columns=colnames)
            step_history['level'] = range(env.n_levels)
            option_history = pd.concat([option_history, step_history])
        option_history_long = pd.melt(option_history, id_vars=["trial", "step", "level"], var_name="action")
        option_history_long['action'] = pd.to_numeric(option_history_long['action'])
        option_history_long.to_csv(self.data_path + "/option_history_long_e" + str(self.env_id) + "_a" + str(self.agent_id) + ".csv")

    def save_rest(self, env, which_data):
        # for which_data = ['state', 'event', 'v', 'n']!
        data = getattr(self, which_data)
        colnames = [str(i) for i in range(env.n_basic_actions)]
        data_hist = pd.DataFrame(columns=colnames)
        for trial in range(data.shape[0]):
            trial_hist = pd.DataFrame(data[trial, :, :], columns=colnames)
            trial_hist['trial'] = trial
            trial_hist['level'] = range(env.n_levels)
            data_hist = pd.concat([data_hist, trial_hist])
        data_hist_long = pd.melt(data_hist, id_vars=["trial", "level"], var_name="action")
        data_hist_long['action'] = pd.to_numeric(data_hist_long['action'])
        data_hist_long.to_csv(self.data_path + "/" + which_data + "_hist_long_e" + str(self.env_id) + "_a" + str(self.agent_id) + ".csv")

    # def save_e(self, env):
    #     colnames = [str(i) for i in range(env.n_basic_actions)]
    #     e_history = pd.DataFrame(columns=colnames)
    #     for row in range(self.e.shape[0]):
    #         for option in range(self.e.shape[1]):
    #             option_e_history = pd.DataFrame(self.e[row, option, :, :], columns=colnames)
    #             option_e_history['option'] = option
    #             option_e_history['action'] = range(env.n_basic_actions)
    #             option_e_history['trial'] = row
    #             e_history = pd.concat([e_history, option_e_history])
    #     e_history.head()
    #     e_history_long = pd.melt(e_history, id_vars=["trial", "option", "action"], var_name="feature")
    #     e_history_long['feature'] = pd.to_numeric(e_history_long['feature'])
    #     e_history_long.to_csv(self.data_path + "/e_history_long_e" + str(self.env_id) + "_a" + str(self.agent_id) + ".csv")

    # Update histories
    def update_theta_history(self, agent, option, trial):
        self.theta[self.theta_row, :, :, :agent.theta.n_basic_actions] = agent.theta.get()
        self.theta[self.theta_row, :, :, -2] = trial
        self.theta[self.theta_row, :, :, -1] = agent.theta.option_coord_to_index(option)
        self.theta_row += 1

    def update_option_history(self, agent, trial):
        self.step = 0
        for option in agent.option_stack:  # list all current options
            self.option[self.option_row, option[0], option[1]] = 1
            self.option[self.option_row, :, -2] = trial
            self.option[self.option_row, :, -1] = self.step
            self.step += 1
            self.option_row += 1
