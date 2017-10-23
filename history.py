import numpy as np
import pandas as pd
import os


class History(object):
    def __init__(self, env, envi, agent, ag):
        self.envi = envi
        self.ag = ag
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

    @staticmethod
    def get_data_path(agent, env, data_dir):
        hier = 'hierarchical' if agent.hier_level > 0 else 'flat'
        folder_structure = "/n_options_per_level_" + str(env.n_options_per_level) +\
                           "/option_length_" + str(env.option_length) +\
                           "/" + hier +\
                           "/" + agent.learning_signal +\
                           "/alpha_" + str(agent.alpha) +\
                           "/e_lambda_" + str(agent.e_lambda) +\
                           "/n_lambda_" + str(agent.n_lambda) +\
                           "/gamma_" + str(agent.gamma) +\
                           "/epsilon_" + str(agent.epsilon) +\
                           "/distraction_" + str(agent.distraction) + "/"
        data_path = data_dir + folder_structure
        return data_path

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

    def save_all(self, data_dir, env, agent):
        data_path = self.get_data_path(agent, env, data_dir)
        if not os.path.isdir(data_path):
            os.makedirs(data_path)
        # self.save_e(env, data_path)
        self.save_events(env, data_path)
        self.save_states(env, data_path)
        self.save_v(env, data_path)
        self.save_n(env, data_path)
        self.save_theta(env, data_path)
        self.save_options(env, data_path)

    def save_e(self, env, data_path):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        e_history = pd.DataFrame(columns=colnames)
        for row in range(self.e.shape[0]):
            for option in range(self.e.shape[1]):
                option_e_history = pd.DataFrame(self.e[row, option, :, :], columns=colnames)
                option_e_history['option'] = option
                option_e_history['action'] = range(env.n_basic_actions)
                option_e_history['trial'] = row
                e_history = pd.concat([e_history, option_e_history])
        e_history.head()
        e_history_long = pd.melt(e_history, id_vars=["trial", "option", "action"], var_name="feature")
        e_history_long['feature'] = pd.to_numeric(e_history_long['feature'])
        e_history_long.to_csv(data_path + "/e_history_long_e" + str(self.envi) + "_a" + str(self.ag) + ".csv")

    def save_theta(self, env, data_path):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        colnames = colnames + ['trial', 'updated_option']
        theta_history = pd.DataFrame(columns=colnames)
        for row in range(self.theta.shape[0]):
            for option in range(self.theta.shape[1]):
                option_theta_history = pd.DataFrame(self.theta[row, option, :, :], columns=colnames)
                option_theta_history['option'] = option
                option_theta_history['action'] = range(env.n_basic_actions)
                theta_history = pd.concat([theta_history, option_theta_history])
        theta_history.head()
        theta_history = theta_history[theta_history['1'] != 0]
        theta_history_long = pd.melt(theta_history, id_vars=["trial", "option", "action", "updated_option"],
                                     var_name="feature")
        theta_history_long['feature'] = pd.to_numeric(theta_history_long['feature'])
        theta_history_long.to_csv(data_path + "/theta_history_long_e" + str(self.envi) + "_a" + str(self.ag) + ".csv")

    def save_options(self, env, data_path):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        colnames = colnames + ['trial', 'step']
        option_history = pd.DataFrame(columns=colnames)
        for row in range(self.option.shape[0]):
            step_history = pd.DataFrame(self.option[row, :, :], columns=colnames)
            step_history['level'] = range(env.n_levels)
            option_history = pd.concat([option_history, step_history])
        option_history.head()
        option_history_long = pd.melt(option_history, id_vars=["trial", "step", "level"], var_name="action")
        option_history_long['action'] = pd.to_numeric(option_history_long['action'])
        option_history_long.to_csv(data_path + "/option_history_long_e" + str(self.envi) + "_a" + str(self.ag) + ".csv")

    def save_states(self, env, data_path):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        state_history = pd.DataFrame(columns=colnames)
        for trial in range(self.state.shape[0]):
            trial_state_history = pd.DataFrame(self.state[trial, :, :], columns=colnames)
            trial_state_history['trial'] = trial
            trial_state_history['level'] = range(env.n_levels)
            state_history = pd.concat([state_history, trial_state_history])
        state_history_long = pd.melt(state_history, id_vars=["trial", "level"], var_name="action")
        state_history_long['action'] = pd.to_numeric(state_history_long['action'])
        state_history_long.to_csv(data_path + "/state_history_long_e" + str(self.envi) + "_a" + str(self.ag) + ".csv")

    def save_events(self, env, data_path):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        event_history = pd.DataFrame(columns=colnames)
        for trial in range(self.event.shape[0]):
            trial_event_history = pd.DataFrame(self.event[trial, :, :], columns=colnames)
            trial_event_history['trial'] = trial
            trial_event_history['level'] = range(env.n_levels)
            event_history = pd.concat([event_history, trial_event_history])
        event_history_long = pd.melt(event_history, id_vars=["trial", "level"], var_name="action")
        event_history_long['action'] = pd.to_numeric(event_history_long['action'])
        event_history_long.to_csv(data_path + "/event_history_long_e" + str(self.envi) + "_a" + str(self.ag) + ".csv")

    def save_v(self, env, data_path):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        v_history = pd.DataFrame(columns=colnames)
        for trial in range(self.v.shape[0]):
            trial_v_history = pd.DataFrame(self.v[trial, :, :], columns=colnames)
            trial_v_history['trial'] = trial
            trial_v_history['level'] = range(env.n_levels)
            v_history = pd.concat([v_history, trial_v_history])
        v_history.head()
        v_history_long = pd.melt(v_history, id_vars=["level", "trial"], var_name="action")
        v_history_long['action'] = pd.to_numeric(v_history_long['action'])
        v_history_long.to_csv(data_path + "/v_history_long_e" + str(self.envi) + "_a" + str(self.ag) + ".csv")

    def save_n(self, env, data_path):
        colnames = [str(i) for i in range(env.n_basic_actions)]
        n_history = pd.DataFrame(columns=colnames)
        for trial in range(self.n.shape[0]):
            trial_n_history = pd.DataFrame(self.n[trial, :, :], columns=colnames)
            trial_n_history['trial'] = trial
            trial_n_history['level'] = range(env.n_levels)
            n_history = pd.concat([n_history, trial_n_history])
        n_history.head()
        n_history_long = pd.melt(n_history, id_vars=["level", "trial"], var_name="action")
        n_history_long['action'] = pd.to_numeric(n_history_long['action'])
        n_history_long.to_csv(data_path + "/n_history_long_e" + str(self.envi) + "_a" + str(self.ag) + ".csv")
