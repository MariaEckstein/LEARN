plot_each_trial = function(all_dat, gg_save, plot_dir) {

  for (tr in 0:max(all_dat$event_hist$trial)) {
    
    # Events/actions in each trial
    gg_event = ggplot(subset(all_dat$event_hist, trial == tr), aes(action, factor(level))) +
      geom_tile(aes(fill = value), color = "white") +
      geom_text(aes(label = round(value, 2)), size = 1.5) +
      labs(x = "Action ID", y = "Level") +
      scale_fill_gradient(low = "black", high = "yellow", na.value = "grey") +
      theme(legend.position = "none") +
      facet_grid(~ factor(trial, levels = unique(all_dat$v_hist$trial), labels = paste("Trial", unique(all_dat$v_hist$trial))))
    
    # State in each trial
    gg_state = gg_event
    gg_state$data = subset(all_dat$state_hist, trial == tr)
    gg_state = gg_state +
      scale_fill_gradient(low = "black", high = "yellow")
    
    # Options in each trial
    gg_option = gg_event
    gg_option$data = subset(all_dat$option_hist, trial == tr)
    gg_option = gg_option +
      scale_fill_gradient(low = "white", high = "steelblue") +
      facet_wrap(~ step)
    
    # Values in each trial
    upper_limit = ifelse(learning_signal == "novelty", 1, max(all_dat$v_hist$level)) 
    gg_value = gg_event
    gg_value$data = subset(all_dat$v_hist, trial == tr)
    gg_value = gg_value +
      scale_fill_gradient(low = "white", high = "red", limits = c(0, upper_limit))  # make each plot based on the same color
    
    # Novelty in each trial
    gg_novel = gg_event
    gg_novel$data = subset(all_dat$n_hist, trial == tr)
    gg_novel = gg_novel +
      scale_fill_gradient(low = "white", high = "red", limits = c(0, upper_limit))
    
    # Thetas in each trial
    for (op in unique(all_dat$theta_hist$option)) {
      
      theta_dat = subset(all_dat$theta_hist, option == op & updated_option == op & trial == tr)
      if (nrow(theta_dat) > 0) {
        gg_theta = gg_value
        gg_theta$data = theta_dat
        gg_theta = gg_theta +
          aes(action, feature) +
          scale_fill_gradient(low = "white", high = "red", limits = c(min(all_dat$theta_hist$value), max(all_dat$theta_hist$value)))
        
        # Put theta-option graphs together
        if (hier == "hierarchical") {
          grob_theta_option = grid.arrange(gg_theta, gg_option, gg_event, ncol = 1)  # , gg_e
        } else {  # flat agents don't have options
          grob_theta_option = grid.arrange(gg_theta, gg_state, ncol = 1)
        }
        
        file_path = file.path(plot_dir, paste("grob_theta_option", op))
        if (!dir.exists(file_path)) {
          dir.create(file_path)
        }
        if (gg_save) {
          file_name = file.path(file_path, paste("trial", tr, ".png", sep = ""))
          print(file_name)
          ggsave(file_name, grob_theta_option, width=max(all_dat$event_hist$action), height=max(all_dat$event_hist$level))
        }
      }
    }
    
    # Put value-option-event graphs together
    if (hier == "hierarchical") {
      grob_event_value = grid.arrange(gg_value, gg_novel, gg_option, gg_event, ncol = 2)
    } else {
      grob_event_value = grid.arrange(gg_value, gg_novel, gg_event, ncol = 2)
    }
    
    file_path = file.path(plot_dir, paste("grob_event_value"))
    if (!dir.exists(file_path)) {
      dir.create(file_path)
    }
    if (gg_save) {
      ggsave(file.path(file_path, paste("trial", tr, ".png", sep = "")), grob_event_value, width=max(all_dat$event_hist$action), height=max(all_dat$event_hist$level))
    }
  }
  
  
  
  # Plot actions options events and values over time
  # Events, selected trials
  gg_event_multi = gg_event
  gg_event_multi$data = subset(all_dat$event_hist, trial < 10 | trial > (max(trial) - 10))
  gg_event_multi = gg_event_multi + facet_wrap(~ trial)
  
  # Options, selected trials
  gg_option_multi = gg_option
  gg_option_multi$data = subset(all_dat$option_hist, trial < 10 | trial > (max(trial) - 10))
  gg_option_multi = gg_option_multi + facet_wrap(~ trial)
  
  # Values, selected trials
  gg_value_multi = gg_value
  gg_value_multi$data = subset(all_dat$v_hist, trial < 10 | trial > (max(trial) - 10))
  gg_value_multi = gg_value_multi + facet_wrap(~ trial)
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_event_multi.png"), gg_event_multi, width=6, height=4)
    ggsave(file.path(plot_dir, "gg_option_multi.png"), gg_option_multi, width=6, height=4)
    ggsave(file.path(plot_dir, "gg_value_multi.png"), gg_value_multi, width=6, height=4)
  }
  
  # Plot novelty values over time
  gg_summary_novelty = ggplot(all_dat$v_hist, aes(trial, value, color = factor(action), group = level)) +
    stat_summary(fun.data = "mean_se", geom = "pointrange") +
    stat_summary(fun.y = "mean", geom = "line", color = "black") +
    geom_line(aes(group = action)) +
    coord_cartesian(y=c(0, upper_limit), x=c(0, 350)) +
    theme(legend.position = "none") +
    facet_grid(~ factor(level, levels = unique(all_dat$v_hist$level), labels = paste("Level", unique(all_dat$v_hist$level))))
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_summary_novelty.png"), gg_summary_novelty, width = 6, height = 3)
  }
  
  # Plot option features over time
  sum_thet_dat = subset(all_dat$theta_hist, option %in% 0:5 & feature %in% 0:5)
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
  th0 = subset(all_dat$theta_hist, option == 0 & feature %in% 0:7 & action %in% 0:7)
  
  tiles_opt = expand.grid(feature = 0:max(all_dat$theta_hist$feature), option = 0:max(all_dat$theta_hist$option), action = max(all_dat$theta_hist$action))
  tiles = subset(all_dat$theta_hist, trial == max(th0$trial))  # & option %in% 0:2 & action %in% 0:9
  gg_final_policies = ggplot(tiles, aes(feature, factor(action))) +
    geom_tile(aes(fill = value), colour = "white") +
    scale_fill_gradient(low = "white", high = "steelblue") +
    labs(x = "Feature", y = "ActionID") +
    facet_grid(~ factor(option, labels = paste("Option", unique(tiles$option))))
  
  if (gg_save) {
    ggsave(file.path(plot_dir, "gg_final_policies.png"), gg_final_policies, width = 25, height = 3)
  }
  
}