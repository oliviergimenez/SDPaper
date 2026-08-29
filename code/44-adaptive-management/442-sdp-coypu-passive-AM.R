#############################################################
# Passive adaptive management for coypu (ragondin) regulation
# -----------------------------------------------------------
# Goal:
#   Find an optimal control policy u*(N) under structural uncertainty using the 
#   weighted average approach for solving passive adaptive management problems.
#
# State:
#   N = total abundance (1D), discretized onto a finite grid
#
# Action:
#   u in [0,1] = control effort (dimensionless), discretized onto a finite grid
#
# Stochastic dynamics (1-step, e.g. yearly):
#   N_{t+1} = N_t(1 + R(N_t)) - q(u) \times N + \epsilon
#   R(N_t) = r \times (1 - (N_t / K) ^ m)
#   where q(u) is a saturating removal fraction, R(N_t) is the per-capita growth 
#   rate, m describes whether the per-capita growth rate is linear, concave, or 
#   convex, and epsilon is log-normally distributed error
#
# Structural uncertainty
#   We consider a model set with uncertainty about the shape of the per-capita 
#   growth rate, R(N_t), as described by the value of m. 
#   If m = 1, the per-capita growth rate declines linearly as a N approaches K.
#   If m > 1, most density—dependent change occurs at high population levels 
#   (close to K).
#   If m < 1, most density-dependent change occurs at low population levels.
#
# Reward:
#   Reward = -(damage(N) + cost(u) + penalty(N))
#
# Solution:
#   Infinite-horizon discounted value iteration at time t, given belief state 
#   b_t. Solution is simulated for 20 years given known dynamics to demonstrate 
#   updated belief state and improved reward over time.
#
# Notes:
#   Code is adapted from https://github.com/boettiger-lab/mdplearning.
###############################################################################

library(tidyverse)
library(MDPtoolbox)

source("../SDPaper/code/44-adaptive-management/AM_utils.R")

# -----------------------------#
# 1) State and action grids
# -----------------------------#

# Carrying capacity: upper bound defining the support for N
K <- 2000

# Discrete state space for abundance N:
#  - Here step = 20, so we have 101 states from 0 to 2000.
#  - Trade-off: finer step => more precise but slower.
seq_N <- seq(0, K, by = 20)
S <- length(seq_N)  # number of states

# Discrete action space for control effort u in [0,1]:
#  - Here step = 0.05, so we have 21 effort levels.
seq_u <- seq(0, 1, by = 0.05)
A <- length(seq_u)  # number of actions

# Structural uncertainty
#  - Uncertainty is incorporated as an unknown per-capita growth rate, which 
#    takes on different functional forms based on the value of m.
#  - Here we consider a model set m = {0.4, 0.7, 1, 1.3, 1.7}. 
#  - The true system dynamics are represented by m = 1.15, which is bounded, yet 
#    not contained by the model set.
m <- c(0.4, 0.7, 1, 1.15, 1.3, 1.7)
true_m <- m[4]
model_set <- m[!m %in% true_m]

# -----------------------------#
# 2) Parameters 
# -----------------------------#

# Intrinsic growth rate (logistic dynamics): controls growth at low N
r <- 0.9

# Log-standard deviation in log-normally distributed error in population growth
sigma <- 0.1

# Controls how quickly the removal fraction increases with effort u
# Larger alpha => smaller u already removes a lot (stronger response)
alpha <- 1.5

# Tolerable threshold (soft constraint): above this, penalties apply
N_tol <- 400

# -----------------------------#
# 3) Helper functions
# -----------------------------#

# Clamp a number into [lo, hi] to avoid leaving state support
clamp <- function(x, lo, hi) pmin(pmax(x, lo), hi)

# Project a continuous value onto a discrete grid by nearest neighbor
# (this is the "state discretization" step).
round_to_grid <- function(x, grid) {
  grid[which.min((grid - x) ^ 2)]
}

# Removal fraction as a saturating function of effort u:
#   q(u) = 1 - exp(-alpha*u)
# Properties:
#   q(0)=0, increasing, saturating (diminishing returns).
q_u <- function(u) 1 - exp(-alpha * u)

# -----------------------------#
# 4) Population dynamics (1-step)
# -----------------------------#
# Stochastic update:
#   N_{t+1} = N_t(1 + R(N_t)) - q(u) \times N + \epsilon
#   R(N_t) = r \times (1 - (N_t / K) ^ m)
#   where q(u) is a saturating removal fraction, R(N_t) is the per-capita growth 
#   rate, m describes whether the per-capita growth rate is linear, concave, or 
#   convex, and epsilon is log-normally distributed error
#
# Interpretation:
# - logistic growth adds recruits at low N and slows near K, and m describes the 
#   shape of the relationship between the per-capita growth rate, R(N), and N
# - control removes a fraction q(u) of the population
step_fun <- function(N, u, m, sigma) {
  
  Nnext <- N + r * N * (1 - (N / K) ^ m) - q_u(u) * N
  clamp(Nnext, 0, K)
  
  if (Nnext <= 0) {
    x <- c(1, rep(0, S - 1))
  } else { # stochastic
    # P(S | s', sigma)
    x <- dlnorm(seq_N, log(Nnext), sdlog = sigma)
    # normalize
    x <- x / sum(x) 
  }
  return(x)
}

# -----------------------------#
# 5) Reward function = negative total costs
# -----------------------------#
# Components:
# - damage(N): increasing convex function of N (impacts accelerate with abundance)
# - cost(u): increasing convex function of u (marginal effort becomes harder/costlier)
# - penalty(N): soft constraint if N > N_tol (risk aversion / unacceptable state)
reward_fun <- function(N, u) {
  
  # Damages: linear + quadratic
  # - linear = baseline nuisance
  # - quadratic = rapidly increasing impacts at high N
  damage <- 0.1 * N + 0.001 * N ^ 2
  
  # Management cost: convex in u
  # - proportional part + quadratic part for increasing marginal costs
  cost <- 500 * u + 1000 * u ^ 2
  
  # Penalty above threshold: increases linearly beyond N_tol
  penalty <- if (N > N_tol) 2000 * (N - N_tol) / N_tol else 0
  
  # Reward = negative costs (maximize reward <=> minimize cost)
  -(damage + cost + penalty)
}


# -----------------------------#
# 6) Build transition kernels P for each model k
# -----------------------------#
# - P: [S, S, A]_k transition probability array for model k
#      P[s, s_next, a]_k = Prob(next_state = s_next | current_state = s, action = a, model = k)
#


# function for calculating the transition probability array for model k 
# (i.e., given value of m)
get_P <- function(m, S, A, seq_N, seq_u, sigma) {
  
  P <- array(0, dim = c(S, S, A))
  
  for (s in seq_len(S)) {
    
    # Current abundance for state index s
    N <- seq_N[s]
    
    for (a in seq_len(A)) {
      
      # Current effort for action index a
      u <- seq_u[a]
      
      # Continuous next abundance under dynamics + control
      P[s, , a] <- step_fun(N, u, m, sigma)
      
    }
  }
  
  return(P)
}

# create a list of transition matrices for each model k in the model set
P <- list()
for (i in seq_len(length(m))) {
  P[[i]] <- get_P(m[i], S, A, seq_N, seq_u, sigma)
}
names(P) <- c("1", "2", "3", "true", "4", "5")

# -----------------------------#
# 7) Build reward matrix R
# -----------------------------#
# - R: [S, A] reward matrix
#      R[s, a] = immediate reward when choosing action a in state s
#

# get reward
R <- matrix(0, nrow = S, ncol = A)
for (s in seq_len(S)) {
  
  # Current abundance for state index s
  N <- seq_N[s]
  
  for (a in seq_len(A)) {
    
    # Current effort for action index a
    u <- seq_u[a]
    
    # Immediate reward at (N,u)
    R[s, a] <- reward_fun(N, u)
  }
}

# -----------------------------#
# 8) Solve infinite-horizon discounted SDP with known dynamics
# -----------------------------#
# For comparison, first solve the SDP when the system dynamics are known.
# Use value iteration algorithm in MDPtoolbox.
#

# set discount factor
discount <- 0.95

# create list of solutions for known dynamics
known_sol <- list()
known_policies <- data.frame(m = NULL, N = NULL, policy = NULL)
for (i in 1:length(P)) {
  known_sol[[i]] <- mdp_value_iteration(P[[i]], R, 
                                         discount = discount, epsilon = 1e-6)
  known_policies <- rbind(known_policies,
                          data.frame(m = rep(m[[i]], S),
                                     N = seq_N,
                                     policy = known_sol[[i]]$policy))
}

# save
write.csv(known_policies, "../SDPaper/code/44-adaptive-management/data/passive_known_dynamics.csv")

# -----------------------------#
# 9) Solve passive adaptive management with unknown dynamics
# -----------------------------#
# Uses two functions found in AM_utils.R:
# - mdp_compute_policy: calculates the optimal solution, given the transition 
#   dynamics of the model set, P_k, and the belief state using value 
#   iteration
# - bayes_update_model_belief: updates the belief state, b_{t+1}, given the 
#   current belief state, b_t, the states at t and t+1, the action at t, and 
#   the transition dynamics of the model set, P_k

# set constants
Tmax <- 30 # number of time steps for simulation
x0 <- 41 # state at t = 0, (index of seq_N)
a0 <- 1 # action at t = 0

# create model set: list of transition dynamics, P_k, excluding the true system 
# dynamics
model_set <- P[c(1:3, 5, 6)]
true_transition <- P[[4]]
n_models <- length(model_set)

# create vectors to record the state, action, and value at each time step
state <- action <- value <- numeric(Tmax + 1)
state[2] <- x0
action[1] <- a0
time <- 2:(Tmax + 1)

# create a matrix to store the updated belief state over time
# we assume a uniform prior (i.e., each model in the model set is equally 
# probable)
model_prior <- rep(1, n_models) / n_models
belief <- array(NA, dim = c(Tmax + 2, n_models))
belief[2, ] <- model_prior

set.seed(123)

for (t in time){
  
  # calculate the state-dependent policy, given the current belief state
  out <- mdp_compute_policy(transition = model_set, reward = R, discount, 
                            belief[t, ])
  
  # get the optimal action, given the current state
  action[t] <- out$policy[state[t]]
  
  # get the reward, given the current state and action
  value[t] <- R[state[t], action[t]] * discount ^ (t - 1)
  
  # simulate s_{t+1}, given the true system dynamics
  state[t + 1] <- sample(1:S, 1, 
                         prob = true_transition[state[t], , action[t]])
  
  # update the belief state, given the belief at t, s_t, s_{t+1}, a_t, and the 
  # transition dynamics of the model set, P_k
  belief[t + 1, ] <- bayes_update_model_belief(belief[t,], state[t],
                                               state[t + 1], action[t],
                                               model_set)
  
}

# save change in belief state over time
posterior <- as.data.frame(belief[time, ])
colnames(posterior) <- m[c(1:3, 5, 6)]
write.csv(posterior, "../SDPaper/code/44-adaptive-management/data/passive_am_belief.csv")

# save states, actions, and reward over time
df <- data.frame(time = 1:Tmax, state = state[time],
                 action = action[time],
                 value = value[time])
write.csv(df, "../SDPaper/code/44-adaptive-management/data/passive_am_data.csv")

