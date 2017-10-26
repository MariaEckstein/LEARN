read_in_data = function(base_dir, which_data) {
  
  # Set up algo_dat
  algo_dat = list()
  
  # Get data directory
  data_dir = paste(base_dir,
                   "/n_options_per_level_", which_data$n_options_per_level,
                   "/option_length_", which_data$option_length,
                   "/", which_data$hier,
                   "/", which_data$learning_signal,
                   "/alpha_", which_data$alpha,
                   "/e_lambda_", which_data$e_lambda,
                   "/n_lambda_", which_data$n_lambda,
                   "/gamma_", which_data$gamma,
                   "/epsilon_", which_data$epsilon,
                   "/distraction_", which_data$distraction,
                   sep = "")
  file_id = paste("e", which_data$env_id, "_a", which_data$agent_id, sep = "")
  
  # Rules
  file_name = paste("rules_e", which_data$env_id, ".csv", sep = "")
  rules = read.csv(file = file.path(data_dir, file_name), header = T)
  rules$X = NULL
  
  # Event history
  file_name = paste("event_history_long_", file_id, ".csv", sep = "")
  event_hist = read.csv(file = file.path(data_dir, file_name), header = T)
  event_hist$X = NULL
  event_hist = with(event_hist, event_hist[order(trial, level, action),])
  # Add n_disc_events to event_hist
  col_names = c()
  for (level in unique(event_hist$level)) {
    for (action in unique(event_hist$action)) {
      event_occurred = event_hist$value == 1 & event_hist$level == level & event_hist$action == action
      event_occurred_first = min(which(event_occurred))
      if (event_occurred_first < Inf) {
        event_occurred[event_occurred_first:length(event_occurred)] = T
      }
      col_name = paste('l', level, 'a', action, sep = '')
      event_hist[col_name] = event_occurred
      col_names = c(col_names, col_name)
    }
  }
  event_hist$n_disc_events = rowSums(event_hist[col_names])
  
  # Novelty history
  file_name = paste("n_history_long_", file_id, ".csv", sep = "")
  n_hist = read.csv(file = file.path(data_dir, file_name), header = T)
  n_hist$X = NULL
  n_hist$n = n_hist$value
  n_hist$value = exp(-as.numeric(which_data$n_lambda) * n_hist$value)
  
  # Value history
  file_name = paste("v_history_long_", file_id, ".csv", sep = "")
  v_hist = read.csv(file = file.path(data_dir, file_name), header = T)
  v_hist$X = NULL
  if (which_data$hier == "flat") {
    v_hist$value[v_hist$level > 0] = NA
  }
  v_hist = with(v_hist, v_hist[order(trial, level, action),])
  
  # State history
  file_name = paste("state_history_long_", file_id, ".csv", sep = "")
  state_hist = read.csv(file = file.path(data_dir, file_name), header = T)
  state_hist$X = NULL
  state_hist = with(state_hist, state_hist[order(trial, level, action),])
  
  # Option history
  file_name = paste("option_history_long_", file_id, ".csv", sep = "")
  option_hist = read.csv(file = file.path(data_dir, file_name), header = T)
  option_hist$X = NULL
  
  # Theta history
  file_name = paste("theta_history_long_", file_id, ".csv", sep = "")
  theta_hist = read.csv(file = file.path(data_dir, file_name), header = T)
  theta_hist$X = NULL
  theta_hist = subset(theta_hist, value != 0)
  
  # Elig. trace history
  # file_name = paste("e_history_long_", file_id, ".csv", sep = "")
  # e_hist = read.csv(file = file.path(data_dir, file_name, header = T)
  # e_hist$X = NULL
  
  # Add ids to data
  n_hist$hier = which_data$hier
  n_hist$learning_signal = which_data$learning_signal
  n_hist$agent_id = which_data$agent_id
  n_hist$env_id = which_data$env_id
  v_hist$hier = which_data$hier
  v_hist$learning_signal = which_data$learning_signal
  v_hist$agent_id = which_data$agent_id
  v_hist$env_id = which_data$env_id
  state_hist$hier = which_data$hier
  state_hist$learning_signal = which_data$learning_signal
  state_hist$agent_id = which_data$agent_id
  state_hist$env_id = which_data$env_id
  option_hist$hier = which_data$hier
  option_hist$learning_signal = which_data$learning_signal
  option_hist$agent_id = which_data$agent_id
  option_hist$env_id = which_data$env_id
  theta_hist$hier = which_data$hier
  theta_hist$learning_signal = which_data$learning_signal
  theta_hist$agent_id = which_data$agent_id
  theta_hist$env_id = which_data$env_id
  event_hist$hier = which_data$hier
  event_hist$learning_signal = which_data$learning_signal
  event_hist$agent_id = which_data$agent_id
  event_hist$env_id = which_data$env_id
  
  # Put everyting into algo_dat
  algo_dat$rules = rules
  algo_dat$n_hist = n_hist
  algo_dat$v_hist = v_hist
  algo_dat$state_hist = state_hist
  algo_dat$option_hist = option_hist
  algo_dat$theta_hist = theta_hist
  algo_dat$event_hist = event_hist
  
  # Return data
  return(algo_dat)
}