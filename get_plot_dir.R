get_plot_dir = function(which_data, data_dir) {
  
  # Get data directory, create "plots" subfolder
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
  plot_dir = file.path(data_dir, paste("plots_e", which_data$env_id, "a_", which_data$agent_id, sep = ""))
  if (!dir.exists(plot_dir)) {
    dir.create(plot_dir)
  }
  return(plot_dir)
}