get_dir = function(base_dir, which_data) {
  
  dir = list()
  
  # Get data directory
  data_dir = paste(base_dir,
                   # "/n_options_per_level_", which_data$n_options_per_level,
                   # "/option_length_", which_data$option_length,
                   "/", which_data$hier,
                   "/", which_data$learning_signal,
                   # "/alpha_", which_data$alpha,
                   # "/n_lambda_", which_data$n_lambda,
                   # "/gamma_", which_data$gamma,
                   # "/epsilon_", which_data$epsilon,
                   # "/distraction_", which_data$distraction,
                   sep = "")
  
  # Get plot directory and creat if not existent
  plot_dir = file.path(data_dir, paste("plots_e", which_data$env_id, "_a", which_data$agent_id, sep = ""))
  
  # Save into dir list
  dir$data_dir = data_dir
  dir$plot_dir = plot_dir
  
  return(dir)
}