import numpy as np
import itertools


class Environment(object):
    def __init__(self, option_length, n_options_per_level, reward, n_trials, env_id):
        self.id = env_id
        self.n_trials_play = n_trials['play']
        self.n_trials_reward = n_trials['reward']
        self.n_basic_actions = n_options_per_level[0]
        self.n_options_per_level = n_options_per_level
        self.option_length = option_length
        self.n_levels = len(n_options_per_level)
        self.state = np.zeros([self.n_levels, self.n_basic_actions], dtype=bool)  # previous event at each level
        self.events = np.zeros([self.n_levels, self.n_basic_actions], dtype=bool)  # events of the current trial
        self.rewarded_events = np.zeros(self.events.shape)
        for level, event in enumerate(reward['events']):
            self.rewarded_events[level, event] = reward['value']

        # Make the rules for this game
        max_options = max(self.n_options_per_level[1:])
        self.rules = np.full([self.n_levels, max_options, self.option_length], np.nan)
        for level in range(1, self.n_levels):  # Define rules for all higher-level options (= not level 0)
            n_options = self.n_options_per_level[level]
            n_actions = self.n_options_per_level[level-1]
            all_rules = np.array(list(itertools.permutations(range(n_actions), self.option_length)))  # w/out replacem.
            np.random.shuffle(all_rules)
            n_rules = all_rules[:n_options]
            for option in range(n_options):
                self.rules[level, option, :] = n_rules[option]

    def give_rewards(self, external_rewards):
        if external_rewards:
            rewards = self.events * self.rewarded_events  # did a rewarded event occur?
        else:
            rewards = np.zeros(self.events.shape)
        return rewards

    def make_events(self, action, hist, trial):
        self.__make_basic_events(action, hist, trial)
        for level in range(self.n_levels-1):
            self.__make_higher_events(level, hist, trial)

        hist.event[trial, :, :] = self.events.copy()
        return self.events.copy()

    def __make_higher_events(self, level, hist, trial):
        events = hist.event_s[:trial+1, level]
        if trial > self.option_length:
            past_actions = np.append(events[0:-1][~np.isnan(events[0:-1])], events[-1])
            past_n_actions = past_actions[-self.option_length:]
            n_options = self.n_options_per_level[level+1]
            for option in range(n_options):
                option_completed = np.all(past_n_actions == self.rules[level+1, option, :])
                if option_completed:
                    self.state[level+1, :] = 0
                    self.state[level+1, option] = 1
                    self.events[level+1, option] = 1
                    hist.event_s[trial, level+1] = option

    def __make_basic_events(self, action, hist, trial):
        self.state[0, :] = 0
        self.state[0, action[1]] = 1
        self.events[:] = 0
        self.events[0, action[1]] = 1
        hist.event_s[trial, 0] = action[1]
        hist.state[trial, :, :] = self.state.copy()