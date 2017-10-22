import numpy as np
import itertools


class Environment(object):
    def __init__(self, env_stuff, n_trials):
        self.n_trials = n_trials
        self.n_basic_actions = env_stuff['n_options_per_level'][0]
        self.n_options_per_level = env_stuff['n_options_per_level']
        self.option_length = env_stuff['option_length']
        self.n_levels = len(env_stuff['n_options_per_level'])
        self.state = np.zeros([self.n_levels, self.n_basic_actions], dtype=bool)  # previous event at each level
        self.events = np.zeros([self.n_levels, self.n_basic_actions], dtype=bool)  # events of the current trial

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

    def make_events(self, action, hist, trial):
        # Basic-action events (level 0)
        self.state[0, :] = 0
        self.state[0, action[1]] = 1
        self.events[:] = 0
        self.events[0, action[1]] = 1
        hist.event_s[trial, 0] = action[1]
        hist.state[trial, :, :] = self.state.copy()

        # Events on higher levels
        for level in range(self.n_levels-1):
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

        # Update event history
        hist.event[trial, :, :] = self.events.copy()
        return self.events.copy()
