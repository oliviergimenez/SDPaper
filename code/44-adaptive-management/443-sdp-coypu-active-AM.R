#############################################################
# Active adaptive management for coypu (ragondin) regulation
# -----------------------------------------------------------
# Goal:
#   Find an optimal control policy u*(N) under structural uncertainty using an
#   active adaptive management strategy. The problem is formulated as a 
#   partially observable Markov Decision Process (POMDP), where the belief 
#   state enters the state space of the SDP. The POMDP is solved using the 
#   SARSOP (Successive Approximations of the Reachable Space under
#   Optimal Policies) algorithm (Kurniawati et al., 2008).
#
#   We also exemplify the dual control problem by finding the optimal policy
#   for multiple discount factors, representing the way future reward is valued
#   relative to current reward.
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
#   State-dependent optimal action, u*(N), given uncertain model of population 
#   dynamics, initial belief state, and discount factor.
#
# Notes:
#   Kurniawati, H., Hsu, D., & Lee, W. S. (2008, June). Sarsop: Efficient point-
#   based pomdp planning by approximating optimally reachable belief spaces. In 
#   Robotics: Science and systems (Vol. 2008).
###############################################################################

library(tidyverse)
library(sarsop)

# -----------------------------#
# 1) State and action grids
# -----------------------------#

# Carrying capacity: upper bound defining the support for N
K <- 2000

# Discrete state space for abundance N:
#  - Here step = 100, so we have 21 states from 0 to 2000.
#  - Trade-off: finer step => more precise but slower.
seq_N <- seq(0, K, by = 100)
S <- length(seq_N)  # number of states

# Discrete action space for control effort u in [0,1]:
#  - Here step = 0.05, so we have 21 effort levels.
seq_u <- seq(0, 1, by = 0.05)
A <- length(seq_u)  # number of actions

# Structural uncertainty
#  - Uncertainty is incorporated as an unknown per-capita growth rate, which 
#    takes on different functional forms based on the value of m.
#  - Here we consider a model set m = {0.7, 1.3}. 
model_set <- c(0.7, 1.3)
n <- length(model_set)

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
for (i in seq_len(length(model_set))) {
  P[[i]] <- get_P(model_set[i], S, A, seq_N, seq_u, sigma)
}

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
# 8) Set up arguments for POMDP solver (SARSOP algorithm)
# -----------------------------#
# 
#   Note: All below matrix depictions of solver arguments assume five candidate models (n = 5)
#
# - transition: block diagonal matrix that allows belief to be added to state space 
#      - Dimensions: [S * n, S * n, A]  
#      - Each transition matrix in list P for each of n models is a sub matrix
#      - The 0 symbols represent entire blocks filled with zeros that "pad" the 
#        space between the diagonal blocks
#                 __                                        __
#                 | P_{k=1}    0       0       0       0     |
#                 |    0    P_{k=2}    0       0       0     |
#   transition =  |    0       0    P_{k=3}    0       0     |
#                 |    0       0       0    P_{k=4}    0     |
#                 |_   0       0       0       0    P_{k=5} _|
#
# - reward: reward matrix
#      - Dimensions: [S * n, A]  
#      - R represents S x A reward matrix
#             __   __
#             |  R  |
#             |  R  |
#   reward =  |  R  |
#             |  R  |
#             |_ R _|
# 
# - observation: observation matrix, stacked identity matrix to assume perfect observation
#      - Dimensions: [S * n, S, A]  
#      - I refers to identity matrix of dimensions S * S
#      - Rows are the state space (with belief), and columns are the observation space
#                  __   __
#                  |  I  |
#                  |  I  |
#   observation =  |  I  |
#                  |  I  |
#                  |_ I _|
#

## transition
# create an empty transition matrix filled with 0
transition <- array(0, dim = c(S * n, S * n, A))
# create block diagonal matrix by filling in with P
for (i in 1:n) {
  s_start <- (i - 1) * S + 1
  s_end <- S * i
  transition[s_start:s_end, s_start:s_end, ] <- P[[i]]
}

## reward
# create an empty reward matrix
reward <- array(NA, dim = c(S * n, A))
# fill in reward with R
for (i in 1:n) {
  s_start <- (i - 1) * S + 1
  s_end <- S * i
  reward[s_start:s_end, ] <- R
}

## observation
# create an empty observation matrix
observation <- array(NA, dim = c(S * n, S, A))
# fill in with identity matrices
for (i in 1:n) {
  s_start <- (i - 1) * S + 1
  s_end <- S * i
  observation[s_start:s_end, 1:S, ] <- diag(S)
}

# -----------------------------#
# 9) Run POMDP solver (SARSOP algorithm)
# -----------------------------#
#
# Solving the POMDP with the SARSOP algorithm occurs in two steps:
#
# 1. Generate a set of alpha vectors that approximates V_pi(b), or the expected
#    total reward of executing policy pi starting from belief b.
#    - V* is the value function associated with the optimal policy, pi*, and can 
#      be approximated by a piece-wise linear function:
#      V(b) = max(alpha ⋅ b)
#      where alpha is a finite set of vectors called alpha vectors, b is the 
#      discrete vector representation of the belief, and alpha ⋅ b is the inner
#      product
#
# 2. Calculate the policy by selecting the action corresponding to the best
#    alpha-vector at the current belief.
# 
# We will find the optimal policy for two discount factors that represent
# different ways of valuing future reward relative to current reward.

# Set up two discount factors to highlight the dual control problem
discount1 <- 0.95 # high future rewards more valuable
discount2 <- 0.75 # high future rewards less valuable

# Get initial belief (uniform across models)
b <- rep(1, dim(transition)[1]) / dim(transition)[1]

## solve POMDP with higher discount factor
# generate alpha vectors, approximation of V(b)
# Note: this is long-running
time1 <- Sys.time()
alpha1 <- sarsop(
  transition = transition, 
  observation = observation, 
  reward = reward, 
  discount = discount1
)
time2 <- Sys.time()
time2 - time1 # A few minutes later...
# Note: Each alpha-vector is associated with an action.

# compute policy based on alpha vectors
df1 <- compute_policy(alpha1, transition, observation, reward, b)

# save data
write.csv(df1, "../SDPaper/code/44-adaptive-management/data/active_am_data_highdiscount.csv")

## solve POMDP with lower discount factor
# generate alpha vectors, approximation of V(b)
time1 <- Sys.time()
alpha2 <- sarsop(
  transition = transition, 
  observation = observation, 
  reward = reward, 
  discount = discount2
)
time2 <- Sys.time()
time2 - time1 # A few minutes later...

# compute policy based on alpha vectors
df2 <- compute_policy(alpha2, transition, observation, reward, b)

# save data
write.csv(df2, "../SDPaper/code/44-adaptive-management/data/active_am_data_lowdiscount.csv")

