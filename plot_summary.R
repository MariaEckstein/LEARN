plot_summary = function(all_dat, plot_dir) {
  
  # Plot actions options events and values over time
  # Events, selected trials
  gg_event_multi = gg_event
  gg_event_multi$data = subset(all_dat$event_hist, trial < 10 | trial > (max(trial) - 10))
  gg_event_multi = gg_event_multi + facet_wrap(~ trial)
  
  # Options, selected trials
  gg_option_multi = gg_option
  gg_option_multi$data = subset(all_dat$option_history, trial < 10 | trial > (max(trial) - 10))
  gg_option_multi = gg_option_multi + facet_wrap(~ trial)
  
  # Values, selected trials
  gg_value_multi = gg_value
  gg_value_multi$data = subset(all_dat$v_history, trial < 10 | trial > (max(trial) - 10))
  gg_value_multi = gg_value_multi + facet_wrap(~ trial)
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_event_multi.png"), gg_event_multi, width=6, height=4)
    ggsave(file.path(plot_dir, "gg_option_multi.png"), gg_option_multi, width=6, height=4)
    ggsave(file.path(plot_dir, "gg_value_multi.png"), gg_value_multi, width=6, height=4)
  }
  
  # Plot novelty values over time
  gg_summary_novelty = ggplot(all_dat$v_history, aes(trial, value, color = factor(action), group = level)) +
    stat_summary(fun.data = "mean_se", geom = "pointrange") +
    stat_summary(fun.y = "mean", geom = "line", color = "black") +
    geom_line(aes(group = action)) +
    coord_cartesian(y=c(0, upper_limit), x=c(0, 350)) +
    theme(legend.position = "none") +
    facet_grid(~ factor(level, levels = unique(all_dat$v_history$level), labels = paste("Level", unique(all_dat$v_history$level))))
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_summary_novelty.png"), gg_summary_novelty, width = 6, height = 3)
  }
  
  # Plot option features over time
  sum_thet_dat = subset(all_dat$theta_history, option %in% 0:5 & feature %in% 0:5)
  gg_summary_theta = ggplot(sum_thet_dat, aes(trial, value, color = factor(action))) +
    geom_point() +
    labs(color="Action") +
    facet_grid(factor(option, levels=unique(sum_thet_dat$option), labels=paste("Option", unique(sum_thet_dat$option))) ~
                 factor(feature, levels=unique(sum_thet_dat$feature), labels=paste("Feature", unique(sum_thet_dat$feature))))
  
  gg_summary_theta2 = ggplot(sum_thet_dat, aes(trial, value, color = factor(feature))) +
    geom_point() +
    labs(color="Feature") +
    facet_grid(factor(option, levels=unique(sum_thet_dat$option), labels=paste("Option", unique(sum_thet_dat$option))) ~
                 factor(action, levels=unique(sum_thet_dat$action), labels=paste("Action", unique(sum_thet_dat$action))))
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_summary_theta.png"), gg_summary_theta, width = 10, height = 5)
    ggsave(file.path(plot_dir, "gg_summary_theta2.png"), gg_summary_theta2, width = 10, height = 5)
  }
  
  # Plot option values over time}
  th0 = subset(all_dat$theta_history, option == 0 & feature %in% 0:7 & action %in% 0:7)
  
  tiles_opt = expand.grid(feature = 0:max(all_dat$theta_history$feature), option = 0:max(all_dat$theta_history$option), action = max(all_dat$theta_history$action))
  tiles = subset(all_dat$theta_history, trial == max(th0$trial))  # & option %in% 0:2 & action %in% 0:9
  gg_final_policies = ggplot(tiles, aes(feature, factor(action))) +
    geom_tile(aes(fill = value), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(x = "Feature", y = "ActionID") +
    facet_grid(~ factor(option, labels = paste("Option", unique(tiles$option))))
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_final_policies.png"), gg_final_policies, width = 25, height = 3)
  }
}