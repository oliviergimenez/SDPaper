###############################################################################
# 1D Stochastic Dynamic Programming (SDP) / MDP for coypu (ragondin) regulation
# ---------------------------------------------------------------------------
# Goal:
#   Find an optimal control policy u*(N) that trades off:
#     - damages increasing with population size N (e.g., crops, dikes, nuisance)
#     - management costs increasing with control effort u
#     - optional penalty if N exceeds a tolerable threshold N_tol
#
# State:
#   N = total abundance (1D), discretized onto a finite grid
#
# Action:
#   u in [0,1] = control effort (dimensionless), discretized onto a finite grid
#
# Dynamics (1-step, e.g. yearly):
#   N_{t+1} = N_t + r*N_t*(1 - N_t/K) - q(u)*N_t
#   where q(u) is a saturating removal fraction
#
# Reward:
#   Reward = -(damage(N) + cost(u) + penalty(N))
#   (MDPtoolbox maximizes reward, so negative costs => cost minimization)
#
# Solution:
#   Infinite-horizon discounted value iteration (mdp_value_iteration)
#
# Notes:
#   - Transitions are deterministic on the grid because we "snap" N_{t+1}
#     to the nearest grid point (round_to_grid).
#   - You can later add stochasticity by spreading probability mass across
#     nearby grid points instead of using a single next state.
###############################################################################

# -----------------------------#
# 0) Packages + reproducibility
# -----------------------------#
library(MDPtoolbox)
library(tidyverse)
library(patchwork)
set.seed(666)

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

cat("Number of states S =", S, "\n")
cat("Number of actions A =", A, "\n")

# -----------------------------#
# 2) Parameters (toy / to calibrate)
# -----------------------------#

# Intrinsic growth rate (logistic dynamics): controls growth at low N
r <- 0.9

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
  grid[which.min((grid - x)^2)]
}

# Removal fraction as a saturating function of effort u:
#   q(u) = 1 - exp(-alpha*u)
# Properties:
#   q(0)=0, increasing, saturating (diminishing returns).
q_u <- function(u) 1 - exp(-alpha * u)

# Parameter (same as in the model)
alpha <- 1.5

# Effort grid
u <- seq(0, 1, length.out = 200)

# Removal fraction
df <- data.frame(
  u = u,
  q = 1 - exp(-alpha * u)
)

fig1a <- ggplot(df, aes(x = u, y = q)) +
  geom_line(linewidth = 1) +
  #  geom_hline(yintercept = 0.5, linetype = "dashed") +
  labs(
    #title = "Saturating removal function",
    x = "Control effort u",
    y = "Removal fraction q(u)"
  ) +
  theme_minimal() +
  ggtitle("A.")
fig1a

# -----------------------------#
# 4) Population dynamics (1-step)
# -----------------------------#
# Deterministic update:
#   N_{t+1} = N_t + r N_t (1 - N_t/K) - q(u) N_t
#
# Interpretation:
# - logistic growth adds recruits at low N and slows near K
# - control removes a fraction q(u) of the population
step_fun <- function(N, u) {
  Nnext <- N + r * N * (1 - N / K) - q_u(u) * N
  clamp(Nnext, 0, K)
}

# -----------------------------#
# 5) Reward function = negative total costs
# -----------------------------#
# Components:
# - damage(N): increasing convex function of N (impacts accelerate with abundance)
# - cost(u): increasing convex function of u (marginal effort becomes harder/costlier)
# - penalty(N): soft constraint if N > N_tol (risk aversion / unacceptable state)
reward_fun <- function(N, u) {
  
  # Damages (toy): linear + quadratic
  # - linear = baseline nuisance
  # - quadratic = rapidly increasing impacts at high N
  damage <- 0.1 * N + 0.001 * N^2
  
  # Management cost (toy): convex in u
  # - proportional part + quadratic part for increasing marginal costs
  cost <- 50 * u + 200 * u^2
  
  # Penalty above threshold (toy): increases linearly beyond N_tol
  penalty <- if (N > N_tol) 2000 * (N - N_tol) / N_tol else 0
  
  # Reward = negative costs (maximize reward <=> minimize cost)
  -(damage + cost + penalty)
}

# Parameters
N_tol <- 400
u_example <- 0.6

# Abundance grid
N <- seq(0, 2000, length.out = 300)

# Cost components
damage <- 0.1 * N + 0.001 * N^2

penalty <- ifelse(
  N > N_tol,
  2000 * (N - N_tol) / N_tol,
  0
)

damage_plus_penalty <- damage + penalty

control_cost_value <- 50 * u_example + 200 * u_example^2

# Data for plotting
df_cost <- rbind(
  data.frame(
    N = N,
    component = "Damage",
    value = damage
  ),
  data.frame(
    N = N,
    component = "Damage + penalty",
    value = damage_plus_penalty
  )
)

df_cost$component <- factor(
  df_cost$component,
  levels = c("Damage", "Damage + penalty")
)

# Values near the right edge for label placement
x_label <- 1800

y_damage_label <-
  0.1 * x_label +
  0.001 * x_label^2

y_penalty_label <-
  y_damage_label +
  2000 * (x_label - N_tol) / N_tol


fig1b <- ggplot(
  df_cost,
  aes(x = N, y = value, linetype = component)
) +
  
  geom_line(linewidth = 0.9) +
  
  # Control cost
  geom_hline(
    yintercept = control_cost_value,
    linetype = "dotdash",
    linewidth = 0.7
  ) +
  
  # Tolerance threshold
  geom_vline(
    xintercept = N_tol,
    linetype = "dotted",
    linewidth = 0.7
  ) +
  
  # Threshold annotation
  annotate(
    "text",
    x = N_tol + 30,
    y = 5900,
    label = "N[tol] == 400",
    parse = TRUE,
    hjust = 0,
    size = 3
  ) +
  
  # Curve labels
  annotate(
    "text",
    x = x_label,
    y = y_penalty_label + 450,
    label = "Damage + penalty",
    hjust = 1,
    size = 3
  ) +
  
  annotate(
    "text",
    x = x_label,
    y = y_damage_label + 300,
    label = "Damage",
    hjust = 1,
    size = 3
  ) +
  
  annotate(
    "text",
    x = x_label,
    y = control_cost_value + 320,
    label = "Control cost",
    hjust = 1,
    size = 3
  ) +
  
  scale_linetype_manual(
    values = c(
      "Damage" = "solid",
      "Damage + penalty" = "dashed"
    )
  ) +
  
  labs(
    x = "Abundance N",
    y = "Immediate cost"
  ) +
  
  guides(linetype = "none") +
  
  theme_minimal() +
  
  ggtitle("B.")

fig1b

# -----------------------------#
# 6) Build transition kernel P and reward matrix R
# -----------------------------#
# MDPtoolbox expects:
# - P: [S, S, A] transition probability array
#      P[s, s_next, a] = Prob(next_state = s_next | current_state = s, action = a)
# - R: [S, A] reward matrix
#      R[s, a] = immediate reward when choosing action a in state s
#
# Here transitions are deterministic on the grid:
# for each (s,a), we compute N_{t+1} (continuous), snap it to the nearest grid
# state, and assign probability 1 to that next state.
P <- array(0, dim = c(S, S, A))
R <- matrix(0, nrow = S, ncol = A)

for (s in seq_len(S)) {
  
  # Current abundance for state index s
  N <- seq_N[s]
  
  for (a in seq_len(A)) {
    
    # Current effort for action index a
    u <- seq_u[a]
    
    # Continuous next abundance under dynamics + control
    Nnext <- step_fun(N, u)
    
    # Discretize to the nearest grid state
    Ndisc <- round_to_grid(Nnext, seq_N)
    
    # Convert discretized abundance value to a state index
    s_next <- which(seq_N == Ndisc)
    
    # Deterministic transition: prob 1 to s_next
    P[s, s_next, a] <- 1
    
    # Immediate reward at (N,u)
    R[s, a] <- reward_fun(N, u)
  }
}

# Sanity check: dimensions + row sums (each row of P[,,a] sums to 1)
mdp_check(P, R)

# -----------------------------#
# 7) Solve infinite-horizon discounted MDP (value iteration)
# -----------------------------#
# discount must satisfy 0 <= gamma < 1
# epsilon controls convergence tolerance
sol <- mdp_value_iteration(P, R, discount = 0.99, epsilon = 1e-6)

# sol$policy: best action index for each state (length S)
# sol$V: value function V(s)
str(sol)

# -----------------------------#
# 8) Extract optimal policy u*(N) and plot
# -----------------------------#
u_star <- seq_u[sol$policy]

# plot(seq_N, u_star, type = "l",
#      xlab = "Abundance N (state grid)",
#      ylab = "Optimal control effort u*",
#      main = "Optimal regulation policy for coypu (1D MDP)")

# -----------------------------#
# 9) Simulate a trajectory under the optimal policy
# -----------------------------#
# Because transitions are deterministic on the grid, the trajectory is
# deterministic given N0 (after snapping to the nearest grid point).
simulate_policy <- function(N0, n_years = 30) {
  
  N_path <- numeric(n_years)
  u_path <- numeric(n_years)
  reward_path <- numeric(n_years)
  
  # Initial abundance (clamped to [0,K])
  N_path[1] <- clamp(N0, 0, K)
  
  for (t in 1:(n_years - 1)) {
    
    # Find the closest state index (in case N0 is not exactly on the grid)
    s <- which.min((seq_N - N_path[t])^2)
    
    # Optimal action index and corresponding effort
    a <- sol$policy[s]
    u <- seq_u[a]
    
    # Store action and reward at current time
    u_path[t] <- u
    reward_path[t] <- reward_fun(N_path[t], u)
    
    # Step forward under dynamics and snap to grid (consistent with the MDP)
    Nnext <- step_fun(N_path[t], u)
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

#--------------------

# optimal policy u*(N) 
policy_df <- data.frame(
  N = seq_N,
  u_star = seq_u[sol$policy]
)

p_policy <- ggplot(policy_df, aes(x = N, y = u_star)) +
  geom_line(linewidth = 0.9) +
  labs(
    title = "",
    x = "Abundance N",
    y = "Optimal effort u*"
  ) +
  theme_minimal() +
  ggtitle("C.")
p_policy

# trajectory as facetted series (N and u) ----
traj_long <- rbind(
  data.frame(time = traj$year, variable = "Abundance N(t)", value = traj$N, series_type = "line"),
  data.frame(time = traj$year, variable = "Effort u(t)",    value = traj$u, series_type = "step")
)

p_traj <- ggplot(traj_long, aes(x = time, y = value)) +
  geom_line(data = subset(traj_long, series_type == "line"), linewidth = 0.9) +
  geom_step(data = subset(traj_long, series_type == "step"), linewidth = 0.9, linetype = 2) +
  facet_wrap(~ variable, ncol = 1, scales = "free_y") +
  geom_hline(data = data.frame(variable = "Abundance N(t)", y = N_tol),
             aes(yintercept = y), linetype = "dotted") +
  labs(
    title = "",
    x = "Time step",
    y = NULL
  ) +
  theme_minimal() +
  ggtitle("D.")
p_traj

# combine all figures
final_plot <- fig1a + fig1b + p_policy + p_traj + plot_layout(nrow = 2)

ggsave("../SDPaper/figures/figure2.png", final_plot, dpi = 600, height = 6, width = 8)


#--------------------

# -----------------------------#
# 10) Tiny demo: why round_to_grid()?
# -----------------------------#
# Ecological dynamics are continuous (e.g., Nnext = 437.6), but MDP states are
# discrete (e.g., 0, 20, 40, ...). We therefore map continuous outcomes to the
# nearest grid state.
x <- 437.6
grid <- c(400, 420, 440, 460)
cat("Continuous x =", x, "\n")
cat("Rounded to grid =", round_to_grid(x, grid), "\n")

