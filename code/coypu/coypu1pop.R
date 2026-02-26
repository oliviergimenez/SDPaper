###############################################################################
# 1D Stochastic Dynamic Programming (SDP) / MDP for coypu (ragondin) regulation
# ---------------------------------------------------------------------------
# Goal: find an optimal control policy u*(N) that trades off:
#   - damages that increase with population size N (e.g., crops, dikes, nuisance)
#   - management costs that increase with control effort u
#   - optional penalty if N exceeds a tolerable threshold N_tol
#
# State:    N = total abundance (1D)
# Action:   u in [0,1] = control effort / intensity (dimensionless)
# Dynamics: density-dependent growth (logistic) minus removals due to control
# Reward:   negative total cost = -(damage + control_cost + penalty)
#
# Solution: value iteration from MDPtoolbox
###############################################################################

library(MDPtoolbox)

#-----------------------------#
# 1) Define state and action grids
#-----------------------------#

# Carrying capacity: upper bound on population size (also defines state support)
K <- 2000

# Discretized state space for abundance N
# - Choose a step that balances precision vs runtime.
# - With by = 20 and K = 2000 => 101 states, which is lightweight.
seq_N <- seq(0, K, by = 20)
S <- length(seq_N)  # number of states

# Discretized action space for control effort u in [0, 1]
# - by = 0.05 => 21 actions
seq_u <- seq(0, 1, by = 0.05)
A <- length(seq_u)  # number of actions

#-----------------------------#
# 2) Biological + management parameters (toy / to calibrate)
#-----------------------------#

# Intrinsic growth rate (ragondin: can be high; pick something plausible for demo)
# In a logistic model, r around 0.5–1 can produce fast growth at low N.
r <- 0.9

# "alpha" controls how quickly the removal fraction increases with effort u.
# Larger alpha => small u already removes a lot.
alpha <- 1.5

# Threshold abundance above which the situation becomes "unacceptable"
# (e.g., damages explode; managerial target zone)
N_tol <- 400

#-----------------------------#
# 3) Small helper functions
#-----------------------------#

# Clamp a number into [lo, hi] to avoid leaving the state support
clamp <- function(x, lo, hi) pmin(pmax(x, lo), hi)

# Project a continuous next value onto the discrete grid by nearest neighbor.
# This makes the transition deterministic on the grid (no interpolation).
round_to_grid <- function(x, grid) {
  grid[which.min((grid - x)^2)]
}

# Removal fraction as a saturating function of effort u:
#   q(u) = 1 - exp(-alpha * u)
# Properties:
#   q(0)=0, q(1)=1-exp(-alpha), increasing, saturating (diminishing returns).
q_u <- function(u) 1 - exp(-alpha * u)

#-----------------------------#
# 4) Population dynamics (1-year step)
#-----------------------------#
# We specify a simple deterministic model:
#   N_{t+1} = N_t + r*N_t*(1 - N_t/K) - q(u)*N_t
#
# Notes:
# - The logistic term r*N*(1-N/K) gives positive growth at low N and
#   negative growth above K (but we clamp at [0,K]).
# - Control removes a fraction q(u) of N.
# - This is a stylized model: in a more realistic setting, you'd replace
#   this with a demographic model and/or stochasticity.

step_fun <- function(N, u) {
  # expected next abundance (continuous)
  Nnext <- N + r * N * (1 - N / K) - q_u(u) * N
  
  # enforce bounds
  clamp(Nnext, 0, K)
}

#-----------------------------#
# 5) Reward (utility) function
#-----------------------------#
# We set reward as negative total costs, so maximizing reward is minimizing cost.
#
# Components (example choices):
# - damage: increasing convex function of N (quadratic captures fast escalation)
# - cost:   increasing convex function of u (effort is expensive; convex = marginally harder)
# - penalty: additional penalty once N exceeds a tolerable threshold N_tol
#
# You will typically tune these so that:
# - if u is always 0 => N explodes => costs explode
# - if u is always 1 => N collapses but cost is high
# - optimum is a compromise: keep N around some manageable region

reward_fun <- function(N, u) {
  
  # Damages: (toy) linear + quadratic
  # - linear term ~ baseline nuisance
  # - quadratic term ~ accelerating damages at high density
  damage <- 0.1 * N + 0.001 * N^2
  
  # Management cost: convex in effort u
  # - 50*u: baseline proportional cost
  # - 200*u^2: convexity / increasing marginal cost
  cost <- 50 * u + 200 * u^2
  
  # Penalty if above threshold (soft constraint):
  # - zero under N_tol
  # - grows linearly above N_tol (scaled by 2000 here)
  penalty <- if (N > N_tol) 2000 * (N - N_tol) / N_tol else 0
  
  # Reward = negative total costs (because MDPtoolbox maximizes)
  -(damage + cost + penalty)
}

#-----------------------------#
# 6) Build transition array P and reward matrix R
#-----------------------------#
# MDPtoolbox expects:
# - P: 3D array with dimensions [S, S, A]
#      P[s, s_next, a] = Prob( next_state = s_next | current_state = s, action = a )
# - R: matrix [S, A]
#      R[s, a] = reward at state s when choosing action a
#
# Here, because we discretize Nnext by nearest neighbor, the transitions are
# deterministic: for each (s,a) there is exactly one next state with prob 1.

P <- array(0, dim = c(S, S, A))
R <- matrix(0, nrow = S, ncol = A)

for (s in seq_len(S)) {
  
  # current abundance for state s
  N <- seq_N[s]
  
  for (a in seq_len(A)) {
    
    # current action effort
    u <- seq_u[a]
    
    # compute continuous next abundance
    Nnext <- step_fun(N, u)
    
    # discretize to nearest state on the grid
    Ndisc <- round_to_grid(Nnext, seq_N)
    
    # translate Ndisc into a state index
    # (safe here because seq_N is a regular grid and Ndisc is in seq_N)
    s_next <- which(seq_N == Ndisc)
    
    # set deterministic transition
    P[s, s_next, a] <- 1
    
    # compute reward for taking action a in state s
    R[s, a] <- reward_fun(N, u)
  }
}

# Optional sanity check: verifies dimensions and that each row of P sums to 1
mdp_check(P, R)

#-----------------------------#
# 7) Solve for optimal policy (infinite-horizon discounted MDP)
#-----------------------------#
# - discount close to 1 => cares about the long-term
# - epsilon controls convergence tolerance

sol <- mdp_value_iteration(P, R, discount = 0.99, epsilon = 1e-6)

# sol$policy is an integer vector of length S:
#   sol$policy[s] = best action index (1..A) for state s
# sol$V is the value function V(s)

#-----------------------------#
# 8) Extract optimal policy u*(N) and plot
#-----------------------------#

# Optimal effort for each N
u_star <- seq_u[sol$policy]

# Quick base plot (no extra packages)
plot(seq_N, u_star, type = "l",
     xlab = "Abundance N (state)",
     ylab = "Optimal control effort u*",
     main = "Optimal regulation policy for ragondin (1D MDP)")

#-----------------------------#
# 9) (Optional) Simulate a trajectory following the optimal policy
#-----------------------------#
# Because the model is deterministic on the grid, the trajectory is deterministic too.
# This helps sanity-check that the policy does something sensible.

simulate_policy <- function(N0, n_years = 30) {
  
  # storage
  N_path <- numeric(n_years)
  u_path <- numeric(n_years)
  reward_path <- numeric(n_years)
  
  # initialize
  N_path[1] <- clamp(N0, 0, K)
  
  for (t in 1:(n_years - 1)) {
    
    # find the closest state index to current N (in case N0 is off-grid)
    s <- which.min((seq_N - N_path[t])^2)
    
    # optimal action index and value
    a <- sol$policy[s]
    u <- seq_u[a]
    
    # store action and reward
    u_path[t] <- u
    reward_path[t] <- reward_fun(N_path[t], u)
    
    # step forward
    Nnext <- step_fun(N_path[t], u)
    
    # snap to grid (consistent with the MDP)
    N_path[t + 1] <- round_to_grid(Nnext, seq_N)
  }
  
  data.frame(
    year = 1:n_years,
    N = N_path,
    u = u_path,
    reward = reward_path
  )
}

traj <- simulate_policy(N0 = 800, n_years = 40)

# Plot simulated N(t)
plot(traj$year, traj$N, type = "l",
     xlab = "Year",
     ylab = "Abundance N",
     main = "Population trajectory under optimal policy")

# Plot effort u(t)
plot(traj$year, traj$u, type = "s",
     xlab = "Year",
     ylab = "Control effort u",
     main = "Control effort over time under optimal policy")


# Discretizing a Continuous State with round_to_grid()
# 
# 1. Why do we need round_to_grid()?
#   
#   In the ecological model, the population dynamics function (step_fun) returns a continuous value for next year’s abundance:
#   
#   [
#     N_{t+1} = 437.6 \quad \text{(for example)}
#   ]
# 
# However, a Markov Decision Process (MDP) requires the state to belong to a finite set of discrete values (the state grid):
#   
#   [
#     \text{seq_N} = (0,; 20,; 40,; \dots,; 2000)
#   ]
# 
# The role of round_to_grid() is therefore to map the continuous value onto the closest discrete state.
# This step is what connects the ecological model (continuous) to the decision model (discrete).
# 
# 2. The function
# round_to_grid <- function(x, grid) {
#   grid[which.min((grid - x)^2)]
# }
# 3. Step-by-step explanation
# Step 1 — Compute distances to all grid points
# 
# grid - x gives how far the value x is from each grid value.
# 
# Example:
#   
#   x <- 437.6
# grid <- c(400, 420, 440, 460)
# 
# grid - x
# # -37.6  -17.6   2.4  22.4
# Step 2 — Square the distances
# 
# (grid - x)^2 ensures all distances are positive and emphasizes larger deviations.
# 
# (grid - x)^2
# # 1414  309  5.76  501
# Step 3 — Find the smallest distance
# 
# which.min() returns the index of the smallest value (i.e. the closest grid point).
# 
# which.min((grid - x)^2)
# # 3
# Step 4 — Return the corresponding grid value
# grid[3]
# # 440
# 
# So 437.6 is projected onto 440, the nearest discrete state.
# 
# 4. Mathematical interpretation
# 
# The function computes:
#   
#   [
#     \operatorname*{argmin}_{g \in \text{grid}} |x - g|
#   ]
# 
# In words:
#   👉 the grid value that minimizes the distance to x.
# 
# 5. Why this matters in an MDP
# 
# Without this step:
#   
#   The ecological system evolves in a continuous space
# 
# The MDP is defined on a finite state space
# 
# round_to_grid() performs a discretization (or quantization) that allows us to apply dynamic programming.
# 
# 6. Conceptual implications
# 
# Using this projection means:
#   
#   “All population values between two grid points are treated as equivalent.”
# 
# So there is a classic trade-off:
#   
#   Grid resolution	Effect
# Finer grid	More precise policy
# Coarser grid	Faster computation
# 
# This is exactly the standard approximation trade-off in stochastic dynamic programming.
# 
# 7. Equivalent (more readable) version
# round_to_grid <- function(x, grid) {
#   idx <- which.min(abs(grid - x))
#   grid[idx]
# }
# 
# This uses absolute distance instead of squared distance but gives the same result.
# 
# 8. Small numerical examples
# seq_N <- seq(0, 200, by = 20)
# 
# round_to_grid(7, seq_N)     # 0
# round_to_grid(11, seq_N)    # 20
# round_to_grid(89, seq_N)    # 80
# round_to_grid(191, seq_N)   # 200
# 9. Ecological intuition
# 
# You can interpret this as a measurement resolution:
#   
#   A manager does not distinguish between 437 and 442 individuals —
# they think in abundance classes (≈ 440).
# 
# 10. Link to decision theory
# 
# In control and reinforcement learning, this step is known as:
#   
#   state discretization
# 
# state aggregation
# 
# grid-based approximation of the Bellman equation
# 
# It is one of the simplest ways to make continuous ecological dynamics compatible with optimal decision frameworks.

