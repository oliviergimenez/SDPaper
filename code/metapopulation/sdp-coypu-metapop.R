###############################################################################
# SDP / MDP spatial toy model for coypu regulation on a network (occupancy + pop)
# -----------------------------------------------------------------------------
# Goal:
#   Find an optimal *global* management action over a finite horizon (Tmax)
#   that trades off:
#     - damages (proportional to total abundance across sites)
#     - management costs (depending on global action intensity)
#
# System:
#   - n_sites locations connected by a graph
#   - Each site has an occupancy state: 0 = empty, 1 = occupied
#   - If occupied, site has a local abundance (here we keep a simple scalar pop[i])
#
# Approach:
#   1) Define discrete state space: all occupancy patterns {0,1}^n_sites
#   2) For each action a, estimate transition probabilities P_a(s -> s')
#      via Monte Carlo simulation (because the transition is stochastic and
#      depends on local extinction/colonization)
#   3) Define immediate reward R(s,a) = - (damage + control_cost)
#   4) Solve finite-horizon SDP with mdp_finite_horizon()
#   5) Forward simulate trajectories under:
#        - the optimal policy
#        - constant strategies (none/moderate/strong)
#      and compare cumulative costs + dynamics
###############################################################################

# -----------------------------#
# 0) Setup
# -----------------------------#
library(MDPtoolbox)
library(igraph)
library(tidyverse)
library(patchwork)

set.seed(666)

# -----------------------------#
# 1) Network, parameters, states
# -----------------------------#

# Sites (towns) and estimated abundances (2022)
villes  <- c("Candillargues","La Grande-Motte","Lansargues",
             "Marsillargues","Mauguio","Valergues",
             "Mudaison","Saint-Nazaire-de-Pézan")
pop_est <- c(80, 0, 125, 0, 50, 15, 0, 200)
n_sites <- length(villes)

# Build a random dense connected undirected graph
# (toy: ensures connectivity, not based on geography)
g <- sample_gnp(n_sites, p = 0.7)
while (!is_connected(g)) g <- sample_gnp(n_sites, p = 0.7)

# Adjacency matrix (0/1)
adj <- as_adjacency_matrix(g, sparse = FALSE)

# Biological + management parameters (toy values)
K <- 250          # local carrying capacity (used in logistic growth)
r <- 0.7          # intrinsic growth rate (per time step)
actions      <- c(0, 0.1, 0.2)   # harvest fraction: none/moderate/strong
control_cost <- c(0, 5, 10)      # global intervention cost for each action
damage_cost  <- 1                # cost per individual (linear damages)

# Occupancy process parameters
alpha <- 0.15     # extinction slope: higher pop => lower extinction prob
beta  <- 0.1      # colonization per occupied neighbor

# Decision horizon and Monte Carlo settings
Tmax   <- 60
n_sims <- 100

# Discrete state space: all occupancy vectors in {0,1}^n_sites
# state_list is an (n_states x n_sites) matrix, each row = one occupancy pattern
state_list <- as.matrix(expand.grid(replicate(n_sites, c(0,1), simplify = FALSE)))
n_states   <- nrow(state_list)
n_actions  <- length(actions)

# Helper: quickly match an occupancy row to its state index
# We store a character key like "01010101" for each state
state_key <- apply(state_list, 1, paste0, collapse = "")
key_to_id <- setNames(seq_len(n_states), state_key)

# -----------------------------#
# 2) Stochastic occupancy update
# -----------------------------#
# simulate_next_state():
#   Given a current occupancy vector (state) and current abundances (pop),
#   apply local extinction (if occupied) and colonization (if empty) to get
#   next occupancy and updated pops.
#
# Notes:
#   - Extinction: p_ext = exp(-alpha * pop[i])
#       => high abundance -> p_ext small -> persistence more likely
#   - Colonization: depends on number k of occupied neighbors
#       p_col = 1 - (1 - beta)^k  (at least one successful colonization attempt)
#   - If a site becomes colonized, we initialize pop[i] = 1 (toy choice)
simulate_next_state <- function(state, pop, adj) {
  
  next_state <- state
  
  for (i in seq_along(state)) {
    
    if (state[i] == 1) {
      # Occupied site: may go extinct
      p_ext <- exp(-alpha * pop[i])
      if (runif(1) < p_ext) {
        next_state[i] <- 0
        pop[i] <- 0
      }
      
    } else {
      # Empty site: may be colonized from neighbors
      nei <- which(adj[i, ] == 1)
      k   <- sum(state[nei])  # number of occupied neighbors
      
      p_col <- 1 - (1 - beta)^k
      if (runif(1) < p_col) {
        next_state[i] <- 1
        pop[i] <- 1
      }
    }
  }
  
  list(state = next_state, pop = pop)
}

df_ext <- tibble(N = 0:100, p_ext = exp(-0.15*N))
df_col <- tibble(k = 0:5,   p_col = 1 - (1 - 0.1)^k)
p1 <- ggplot(df_ext, aes(N, p_ext)) + geom_line() +
  labs(x = "Local abundance N", y = expression(p[ext](N))) + theme_minimal() +
  ggtitle("A.")
p2 <- ggplot(df_col, aes(k, p_col)) + geom_line() +
  labs(x = "No. occupied neighbors k", y = expression(p[col](k))) + theme_minimal() +
  ggtitle("B.")
p1 + p2

# -----------------------------#
# 3) Monte Carlo transition matrices P and rewards R
# -----------------------------#
# MDPtoolbox expects:
#   - P: list of length n_actions
#        each element is an (n_states x n_states) transition matrix
#   - R: (n_states x n_actions) reward matrix
#
# Here we estimate P_a(s -> s') by:
#   - starting from state s, build pop consistent with occupancy and pop_est
#   - apply deterministic local growth + harvest to update pop
#   - simulate stochastic extinction/colonization n_sims times
#   - tabulate frequencies of next occupancy patterns to approximate probabilities
P <- vector("list", n_actions)
R <- matrix(0, n_states, n_actions)

for (a in seq_len(n_actions)) {
  
  # Initialize transition matrix for action a
  P[[a]] <- matrix(0, n_states, n_states)
  
  for (s in seq_len(n_states)) {
    
    # Current occupancy pattern
    state <- state_list[s, ]
    
    # Build abundance vector: pop_est where occupied, else 0
    pop <- ifelse(state == 1, pop_est, 0)
    
    # ---- deterministic "within-step" population update ----
    # If occupied: logistic growth; then harvest fraction actions[a]
    pop <- pop + r * pop * (1 - pop / K)
    pop <- pmax(0, pop * (1 - actions[a]))
    
    # ---- Monte Carlo sampling of next occupancy ----
    next_states <- matrix(0, nrow = n_sims, ncol = n_sites)
    
    for (m in 1:n_sims) {
      out <- simulate_next_state(state = state, pop = pop, adj = adj)
      next_states[m, ] <- out$state
    }
    
    # Map sampled next states to indices in 1..n_states
    # Faster than nested apply(): use the key trick "0101..."
    idx <- apply(next_states, 1, paste0, collapse = "")
    sid <- unname(key_to_id[idx])
    
    probs <- table(sid) / length(sid)
    
    P[[a]][s, as.integer(names(probs))] <- as.numeric(probs)
    
    # ---- immediate reward (negative immediate cost) ----
    # damage cost proportional to TOTAL abundance after growth+harvest
    # + global control cost for choosing action a
    R[s, a] <- - (damage_cost * sum(pop) + control_cost[a])
  }
}

# Optional sanity check: rows of each P[[a]] should sum to ~1
mdp_check(P, R)

# -----------------------------#
# 4) Solve finite-horizon SDP
# -----------------------------#
disc <- 0.99
res <- mdp_finite_horizon(P, R, discount = disc, N = Tmax)

# res$V      : value function (n_states x (Tmax+1))
# res$policy : optimal action index for each state and time (n_states x Tmax)

# -----------------------------#
# 5) Forward simulation under the optimal policy
# -----------------------------#
# We start from the observed 2022 occupancy pattern implied by pop_est,
# then apply at each time:
#   1) read optimal action from res$policy for current state
#   2) apply growth + harvest on pop
#   3) apply stochastic extinction/colonization
pop_mat   <- matrix(NA, nrow = Tmax, ncol = n_sites)
state_mat <- matrix(NA, nrow = Tmax, ncol = n_sites)

pop   <- pop_est
state <- ifelse(pop > 0, 1, 0)

pop_mat[1, ]   <- pop
state_mat[1, ] <- state

for (t in 1:(Tmax - 1)) {
  
  # find state id for current occupancy
  sid <- key_to_id[paste0(state, collapse = "")]
  
  # optimal action at time t for state sid
  a <- res$policy[sid, t]
  
  # growth + harvest
  pop <- ifelse(state == 1, pop + r * pop * (1 - pop / K), 0)
  pop <- pmax(0, pop * (1 - actions[a]))
  
  # stochastic occupancy update
  out   <- simulate_next_state(state = state, pop = pop, adj = adj)
  state <- out$state
  pop   <- out$pop
  
  pop_mat[t + 1, ]   <- pop
  state_mat[t + 1, ] <- state
}

# -----------------------------#
# 6) Compare cumulative costs: constant vs optimal strategies
# -----------------------------#
state_init <- ifelse(pop_est > 0, 1, 0)
pop_init   <- pop_est

simulate_const <- function(a) {
  pop   <- pop_init
  state <- state_init
  cost  <- numeric(Tmax)
  
  for (t in 1:Tmax) {
    pop <- ifelse(state == 1, pop + r * pop * (1 - pop / K), 0)
    pop <- pmax(0, pop * (1 - actions[a]))
    
    # immediate cost at time t
    cost[t] <- sum(pop) * damage_cost + control_cost[a]
    
    out   <- simulate_next_state(state = state, pop = pop, adj = adj)
    state <- out$state
    pop   <- out$pop
  }
  
  cumsum(cost)
}

cost_none     <- simulate_const(1)
cost_moderate <- simulate_const(2)
cost_strong   <- simulate_const(3)

# Optimal action sequence along the optimal simulated trajectory
action_opt <- sapply(1:Tmax, function(t) {
  sid <- key_to_id[paste0(state_mat[t, ], collapse = "")]
  res$policy[sid, t]
})

cost_step_opt <- map_dbl(seq_len(Tmax), function(t) {
  sum(pop_mat[t, ]) * damage_cost + control_cost[action_opt[t]]
})
cum_cost_opt <- cumsum(cost_step_opt)

df_costs <- tibble(
  Time     = rep(1:Tmax, 4),
  CumCost  = c(cost_none, cost_moderate, cost_strong, cum_cost_opt),
  Strategy = factor(rep(c("None","Moderate","Strong","SDP-Optimal"), each = Tmax),
                    levels = c("None","Moderate","Strong","SDP-Optimal"))
)

p_costs <- ggplot(df_costs, aes(x = Time, y = CumCost, color = Strategy)) +
  geom_line(linewidth = 1.1) +
  labs(x = "Time step", y = "Cumulative cost", color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  ggtitle("C.")
p_costs

# -----------------------------#
# 7) Abundance dynamics per site (optimal vs constant)
# -----------------------------#
simulate_mat <- function(a) {
  pop   <- pop_est
  state <- ifelse(pop_est > 0, 1, 0)
  M <- matrix(0, nrow = Tmax, ncol = length(pop_est))
  
  for (t in 1:Tmax) {
    M[t, ] <- pop
    pop <- ifelse(state == 1, pop + r * pop * (1 - pop / K), 0)
    pop <- pmax(0, pop * (1 - actions[a]))
    out   <- simulate_next_state(state = state, pop = pop, adj = adj)
    state <- out$state
    pop   <- out$pop
  }
  
  M
}

mat_none     <- simulate_mat(1)
mat_moderate <- simulate_mat(2)
mat_strong   <- simulate_mat(3)
mat_opt      <- pop_mat

to_long <- function(M, strategy_label) {
  as_tibble(M) %>%
    set_names(villes) %>%
    mutate(Time = 1:Tmax, Strategy = strategy_label) %>%
    pivot_longer(cols = all_of(villes), names_to = "Site", values_to = "Abundance")
}

df_all <- bind_rows(
  to_long(mat_none, "None"),
  to_long(mat_moderate, "Moderate"),
  to_long(mat_strong, "Strong"),
  to_long(mat_opt, "SDP-Optimal")
) %>%
  mutate(Strategy = factor(Strategy, levels = c("None","Moderate","Strong","SDP-Optimal")))

p_sites <- ggplot(df_all, aes(Time, Abundance, color = Strategy, linetype = Strategy)) +
  geom_line(linewidth = 0.9) +
  facet_wrap(~ Site, ncol = 4) +
  labs(x = "Time step", y = "Abundance", color = NULL, linetype = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  ggtitle("D.")
p_sites

df_tradeoff <- tibble(
  TotalAbundance = seq(0, 800, length.out = 200)
) %>%
  mutate(
    Damage = damage_cost * TotalAbundance,
    None     = control_cost[1],
    Moderate = control_cost[2],
    Strong   = control_cost[3]
  ) %>%
  pivot_longer(
    cols = -TotalAbundance,
    names_to = "Component",
    values_to = "Cost"
  ) %>%
  mutate(
    Component = factor(Component,
                       levels = c("None","Moderate","Strong","Damage"))
  )

p_tradeoff <- ggplot(df_tradeoff,
                     aes(x = TotalAbundance, y = log(Cost), color = Component)) +
  geom_line(linewidth = 1.1) +
  scale_color_manual(
    values = c(
      "None" = "grey40",
      "Moderate" = "orange",
      "Strong" = "red",
      "Damage" = "black"
    )
  ) +
  labs(
    x = "Total abundance across sites",
    y = "Cost per time step (log scale)",
    color = NULL
  ) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  ggtitle("E.")

p_tradeoff

# -----------------------------#
# 8) Global optimal action sequence over time
# -----------------------------#
action_seq <- tibble(
  Time = 1:Tmax,
  Action = factor(action_opt, levels = 1:3, labels = c("None","Moderate","Strong"))
)

p_action <- ggplot(action_seq, aes(x = Time, y = 1, fill = Action)) +
  geom_tile() +
  scale_y_continuous(NULL, breaks = NULL) +
  labs(title = "Global management action over time", x = "Time step", fill = NULL) +
  theme_minimal() +
  theme(panel.grid = element_blank(),
        legend.position = "bottom")
p_action

# -----------------------------#
# 9) Heatmap of removals under the optimal policy
# -----------------------------#

# # Here "removed" is approximated as pop_mat * harvest_fraction at each time.
# harvest_mat <- pop_mat * actions[action_opt]
# 
# df_harvest <- as_tibble(harvest_mat) %>%
#   set_names(villes) %>%
#   mutate(Time = 1:Tmax) %>%
#   pivot_longer(cols = all_of(villes), names_to = "Site", values_to = "Harvested")
# 
# p_harvest <- ggplot(df_harvest, aes(x = Time, y = Site, fill = Harvested)) +
#   geom_tile() +
#   scale_fill_viridis_c(name = "No. removed") +
#   labs(title = "Coypu removals (optimal policy)", x = "Time step", y = "Site") +
#   theme_minimal()
# p_harvest

# Invasion level for each discrete occupancy state: number of occupied sites
occ_count <- rowSums(state_list)  # length = n_states

# Build a long data.frame: (state, time) -> optimal action + invasion level
df_policy <- expand.grid(
  sid  = seq_len(n_states),
  time = seq_len(Tmax)
) %>%
  mutate(
    OccSites = occ_count[sid],
    Action = factor(res$policy[cbind(sid, time)],
                    levels = 1:3,
                    labels = c("None", "Moderate", "Strong"))
  )

ggplot(df_policy, aes(x = time, y = OccSites, fill = Action)) +
  geom_tile() +
  scale_fill_manual(values = c("None" = "grey70", "Moderate" = "orange", "Strong" = "red")) +
  scale_y_continuous(breaks = 0:n_sites) +
  labs(x = "Time step", y = "Number of occupied sites", fill = NULL) +
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "bottom"
  )

# -----------------------------#
# 10) Containment probability: proportion of sites with abundance <= 5
# -----------------------------#
df_summary <- df_all %>%
  group_by(Time, Strategy) %>%
  summarise(perc_low = mean(Abundance <= 5), .groups = "drop")

p_contain <- ggplot(df_summary, aes(Time, perc_low, color = Strategy)) +
  geom_line(linewidth = 1.1) +
  scale_y_continuous(labels = scales::percent) +
  labs(x = "Time step", y = "Proportion of sites", color = NULL) +
  theme_minimal() +
  theme(legend.position = "bottom") +
  ggtitle("E.")
p_contain

# -----------------------------#
# 11) Display or save figures
# -----------------------------#

# combine all figures
final_plot <- p1 + p2 + p_costs + p_sites + p_tradeoff + p_contain + plot_layout(nrow = 2)

ggsave("../SDPaper/figures/figure3.png", final_plot, dpi = 600, height = 6, width = 9)


