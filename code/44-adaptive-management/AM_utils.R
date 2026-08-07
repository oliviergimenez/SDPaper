##############################################################################
##############################################################################

### Auxiliary functions for solving passive and active adaptive management ### 

##############################################################################
##############################################################################
##############################################################################

###############################
###############################
# Passive adaptive management #
###############################
###############################

# calculates the optimal solution, given the transition dynamics of the model 
# set, P_k, and the belief state using value iteration
#
# function arguments
# - transition: list of transition dynamics (i.e., P(s'|s, a) for each model in 
#   the model set
# - reward: the utility matrix U(s,a) of being at state s and taking action a
# - discount: discount factor (1 is no discounting)
# - model_prior: P(k) for each model k
# - max_iter: maximum number of iterations to perform
# - epsilon: convergence tolerance
mdp_compute_policy <- function(transition, reward, discount,
                               model_prior = NULL,
                               max_iter = 500, 
                               epsilon = 1e-5){
  
  # get number of models, states, and actions
  n_models <- length(transition)
  n_states <- dim(transition[[1]])[1]
  n_actions <- dim(transition[[1]])[3]
  
  # create empty vectors for storing V and policies
  next_value <- numeric(n_states)
  next_policy <- numeric(n_states)
  V_model <- array(dim = c(n_states, n_models))
  converged <- FALSE
  t <- 1
  
  # default model prior is uniform distribution
  if (is.null(model_prior)) {
    model_prior <- rep(1, n_models) / n_models
  }
  
  # perform value iteration
  while (t < max_iter && converged == FALSE) {
    Q <- array(0, dim = c(n_states, n_actions))
    for (i in 1:n_actions) {
      for(j in 1:n_models){
        V_model[, j] <- transition[[j]][,,i] %*% next_value
      }
      # Bellman equation: calculate weighted average
      Q[, i] <- reward[, i] + discount * V_model %*% model_prior
    }
    value <- apply(Q, 1, max)
    policy <- apply(Q, 1, which.max)
    
    # check convergence
    if( sum( abs(value - next_value) ) < epsilon ){
      converged <- TRUE
    }
    
    next_value <- value
    next_policy <- policy
    t <- t + 1
    if (t == max_iter)
      message("Note: max number of iterations reached")
  }
  data.frame(state = 1:n_states, policy, value)
}


# updates the belief state, b_{t+1}, given the current belief state, b_t, the 
# states at t and t+1, the action at t, and the transition dynamics of the model 
# set, P_k
#
# function arguments
# - model_prior: P(k) for each model k
# - s_t: state at time t
# - s_t1: state at time t + 1
# - a_t: action at time t
# - transition: list of transition dynamics (i.e., P(s'|s, a) for each model in 
#   the model set

bayes_update_model_belief <- function(model_prior, s_t, s_t1, a_t, transition){
  
  # calculate the number of models
  n_models <- length(transition)
  
  # apply Bayesian theorem to update belief state (i.e., model prior)
  P <- vapply(1:n_models, function(m) transition[[m]][s_t, s_t1, a_t], 
              numeric(1))
  model_prior * P / sum(model_prior * P)
}