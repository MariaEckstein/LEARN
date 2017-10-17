import numpy as np
import itertools
import math


class Environment(object):
    def __init__(self, env_stuff, n_trials):
        self.n_trials = n_trials
        self.n_basic_actions = env_stuff['n_options_per_level'][0]
        self.n_options_per_level = env_stuff['n_options_per_level']
        self.option_length = env_stuff['option_length']
        self.n_levels = len(env_stuff['n_options_per_level'])
        self.state = np.zeros([self.n_levels, self.n_basic_actions], dtype=bool)
        self.events = np.zeros([self.n_levels, self.n_basic_actions], dtype=bool)
        self.past_events = np.full([self.n_trials, self.n_levels], np.nan)

        # Make rules for the game
        max_options = max(self.n_options_per_level[1:])
        self.rules = np.full([self.n_levels, max_options, self.option_length], np.nan)
        for level in range(1, self.n_levels):  # Define rules for all higher-level options (= not level 0)
            n_options = self.n_options_per_level[level]
            n_actions = self.n_options_per_level[level-1]
            all_rules = np.array(list(itertools.combinations(range(n_actions), self.option_length)))  # itertools.combinations draws without replacement
            np.random.shuffle(all_rules)
            n_rules = all_rules[:n_options]
            for option in range(n_options):
                self.rules[level-1, option, :] = n_rules[option]

    def switch_lights(self, action):
        self.state[:] = 0
        self.state[action[0], action[1]] = 1
        return self.state.copy()

    def make_events(self, action, hist, current_trial):
        self.events[:] = 0
        # record basic events
        self.events[action[0], action[1]] = 1
        # self.past_events[current_trial, 0] = action[1]
        # Get event history up to current trial
        for trial in range(current_trial):
            for level in range(self.n_levels):
                event = np.argwhere(hist.event[trial, level, :])
                if len(event) > 0:
                    self.past_events[trial, level] = event

        # check for events on higher levels
        for level in range(self.n_levels-1):
            events = self.past_events[:, level]
            past_n_actions = events[~np.isnan(events)][-self.option_length:]
            n_options = self.n_options_per_level[level+1]
            for option in range(n_options):
                option_completed = np.all(past_n_actions == self.rules[level, option, :])
                if option_completed:
                    self.events[level+1, option] = 1

        return self.events.copy()
