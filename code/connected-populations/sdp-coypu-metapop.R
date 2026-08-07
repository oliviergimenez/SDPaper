###############################################################################
# Worked example 2: Coordinating control across connected populations
# Spatial finite-horizon SDP for coypu management on a network
#
# Main pedagogical idea:
#   - State = occupancy pattern across sites (0/1 at each site)
#   - Actions = None / Moderate / Strong global control
#   - Empty sites may be recolonised from occupied neighbours
#   - Control increases local extinction probability and reduces colonisation
#   - Strong control is effective but expensive
#   - SDP can therefore use strong control when invasion is widespread, then
#     relax to moderate control as the network becomes less invaded
#
# Compared with the previous version:
#   1) the MDP state is now fully consistent with the ecological process:
#      occupancy only (no hidden continuous abundance variable);
#   2) transition probabilities are calculated exactly rather than by Monte Carlo;
#   3) costs are scaled so that Strong is not trivially optimal everywhere;
#   4) strategy comparisons are based on repeated stochastic simulations;
#   5) the main contrast is ecological performance versus management cost.
###############################################################################

# -----------------------------#
# 0) Setup
# -----------------------------#

library(MDPtoolbox)
library(igraph)
library(tidyverse)
library(patchwork)
library(scales)

set.seed(666)

# -----------------------------#
# 1) Sites, network and initial state
# -----------------------------#

# Towns used in the previous coypu abundance analysis
villes <- c(
  "Candillargues",
  "La Grande-Motte",
  "Lansargues",
  "Marsillargues",
  "Mauguio",
  "Valergues",
  "Mudaison",
  "Saint-Nazaire-de-Pézan"
)

# 2022 abundance estimates are used ONLY to define the initial occupancy state.
# The SDP itself is an occupancy model.
pop_est <- c(80, 0, 125, 0, 50, 15, 0, 200)

state_init <- as.integer(pop_est > 0)
n_sites <- length(villes)

# Illustrative connected network.
# This is deliberately schematic and is not intended to reproduce geography.
edges <- matrix(
  c(
    1, 2,
    2, 3,
    3, 4,
    4, 5,
    5, 6,
    6, 7,
    7, 8,
    8, 1,
    1, 3,
    2, 5,
    4, 7,
    6, 8
  ),
  ncol = 2,
  byrow = TRUE
)

g <- graph_from_edgelist(edges, directed = FALSE)
V(g)$name <- villes

adj <- as_adjacency_matrix(g, sparse = FALSE)
adj <- as.matrix(adj)

# -----------------------------#
# 2) Actions and ecological dynamics
# -----------------------------#

action_names <- c("None", "Moderate", "Strong")
n_actions <- length(action_names)

# Local extinction probability for an occupied site.
# Control makes local extinction more likely.
p_ext <- c(
  None     = 0.02,
  Moderate = 0.15,
  Strong   = 0.28
)

# Baseline colonisation probability contributed by each occupied neighbour.
beta_col <- 0.12

# Control also reduces the effective colonisation pressure.
# For example, Strong reduces per-neighbour colonisation pressure by 20%.
col_suppression <- c(
  None     = 0.00,
  Moderate = 0.10,
  Strong   = 0.20
)

# -----------------------------#
# 3) Cost structure
# -----------------------------#

# Damage rises non-linearly with the number of occupied sites.
# The quadratic term represents accelerating impacts as invasion spreads.
damage_linear    <- 1
damage_quadratic <- 4

# Global mobilisation cost + additional cost per occupied site.
# Strong control is deliberately substantially more expensive, creating
# a genuine trade-off between acting aggressively and avoiding unnecessary cost.
control_fixed <- c(
  None     = 0,
  Moderate = 10,
  Strong   = 220
)

control_per_site <- c(
  None     = 0,
  Moderate = 3,
  Strong   = 6
)

damage_fun <- function(n_occ) {
  damage_linear * n_occ +
    damage_quadratic * n_occ^2
}

control_cost_fun <- function(n_occ, a) {
  control_fixed[a] +
    control_per_site[a] * n_occ
}

total_cost_fun <- function(n_occ, a) {
  damage_fun(n_occ) +
    control_cost_fun(n_occ, a)
}

# -----------------------------#
# 4) Decision horizon and state space
# -----------------------------#

Tmax <- 60
disc <- 0.99

# All possible occupancy patterns across 8 sites: 2^8 = 256 states
state_list <- as.matrix(
  expand.grid(replicate(n_sites, c(0, 1), simplify = FALSE))
)

n_states <- nrow(state_list)

state_key <- apply(state_list, 1, paste0, collapse = "")
key_to_id <- setNames(seq_len(n_states), state_key)

state_to_id <- function(state) {
  unname(key_to_id[paste0(state, collapse = "")])
}

occ_count <- rowSums(state_list)

# -----------------------------#
# 5) Transition probabilities
# -----------------------------#

# Probability that site i is occupied at t+1, conditional on the current
# occupancy state and management action.
#
# If occupied:
#   Pr(occupied next time) = 1 - p_ext[action]
#
# If empty:
#   Pr(colonised) = 1 - (1 - beta_eff)^k
# where k is the number of occupied neighbours and beta_eff is reduced by control.

site_occ_prob <- function(state, i, a) {

  if (state[i] == 1) {

    return(1 - p_ext[a])

  } else {

    nei <- which(adj[i, ] == 1)
    k <- sum(state[nei])

    beta_eff <- beta_col * (1 - col_suppression[a])

    return(1 - (1 - beta_eff)^k)
  }
}

# Exact transition probability from current state s to every possible next state.
# Conditional on the current occupancy configuration, local transitions are
# treated as independent. With only 256 states, exact enumeration is transparent
# and avoids Monte Carlo noise in the transition matrices.

transition_row <- function(state, a) {

  p_occ_next <- vapply(
    seq_len(n_sites),
    function(i) site_occ_prob(state, i, a),
    numeric(1)
  )

  probs <- apply(
    state_list,
    1,
    function(next_state) {
      prod(
        ifelse(
          next_state == 1,
          p_occ_next,
          1 - p_occ_next
        )
      )
    }
  )

  probs / sum(probs)
}

# -----------------------------#
# 6) Build transition matrices P and reward matrix R
# -----------------------------#

P <- vector("list", n_actions)
names(P) <- action_names

R <- matrix(
  0,
  nrow = n_states,
  ncol = n_actions,
  dimnames = list(NULL, action_names)
)

for (a in seq_len(n_actions)) {

  P[[a]] <- matrix(0, n_states, n_states)

  for (s in seq_len(n_states)) {

    state <- state_list[s, ]

    P[[a]][s, ] <- transition_row(
      state = state,
      a = a
    )

    n_occ <- sum(state)

    R[s, a] <- -total_cost_fun(
      n_occ = n_occ,
      a = a
    )
  }
}

# Sanity check
mdp_check(P, R)

# -----------------------------#
# 7) Solve the finite-horizon SDP
# -----------------------------#

res <- mdp_finite_horizon(
  P,
  R,
  discount = disc,
  N = Tmax
)

# res$V:
#   optimal value function for every state and time
#
# res$policy:
#   optimal action index for every state and time

# -----------------------------#
# 8) Inspect the optimal policy
# -----------------------------#

# Summarise the proportion of occupancy configurations assigned to each action
# for each number of occupied sites and time step.

df_policy <- expand_grid(
  sid = seq_len(n_states),
  Time = seq_len(Tmax)
) %>%
  mutate(
    OccupiedSites = occ_count[sid],
    Action = factor(
      res$policy[cbind(sid, Time)],
      levels = 1:3,
      labels = action_names
    )
  )

df_policy_summary <- df_policy %>%
  count(Time, OccupiedSites, Action) %>%
  group_by(Time, OccupiedSites) %>%
  mutate(Proportion = n / sum(n)) %>%
  ungroup()

# -----------------------------#
# 9) Stochastic simulation engine
# -----------------------------#

draw_next_state <- function(state, a) {

  sid <- state_to_id(state)

  next_sid <- sample(
    seq_len(n_states),
    size = 1,
    prob = P[[a]][sid, ]
  )

  state_list[next_sid, ]
}

# Simulate one trajectory under a named strategy.
#
# strategy:
#   "None", "Moderate", "Strong", or "SDP"

simulate_strategy <- function(
  strategy,
  seed = NULL
) {

  if (!is.null(seed)) {
    set.seed(seed)
  }

  state <- state_init

  out <- vector("list", Tmax)

  for (t in seq_len(Tmax)) {

    sid <- state_to_id(state)
    n_occ <- sum(state)

    if (strategy == "SDP") {

      a <- res$policy[sid, t]

    } else {

      a <- match(strategy, action_names)
    }

    immediate_damage <- damage_fun(n_occ)
    management_cost <- control_cost_fun(n_occ, a)
    total_cost <- immediate_damage + management_cost

    out[[t]] <- tibble(
      Time = t,
      Strategy = strategy,
      Action = action_names[a],
      OccupiedSites = n_occ,
      Damage = immediate_damage,
      ManagementCost = management_cost,
      TotalCost = total_cost
    )

    state <- draw_next_state(
      state = state,
      a = a
    )
  }

  bind_rows(out) %>%
    mutate(
      CumCost = cumsum(TotalCost),
      CumDamage = cumsum(Damage),
      CumManagementCost = cumsum(ManagementCost)
    )
}

# -----------------------------#
# 10) Compare strategies using repeated simulations
# -----------------------------#

n_rep <- 100

strategies <- c(
  "None",
  "Moderate",
  "Strong",
  "SDP"
)

# Using paired seeds means each strategy is evaluated over the same set of
# replicate numbers. This does not force identical trajectories, because actions
# alter transition probabilities, but makes the simulation experiment reproducible.

sim_all <- map_dfr(
  seq_len(n_rep),
  function(rep_id) {

    map_dfr(
      strategies,
      function(strategy) {

        simulate_strategy(
          strategy = strategy,
          seed = 10000 + rep_id
        ) %>%
          mutate(Replicate = rep_id)
      }
    )
  }
)

sim_all <- sim_all %>%
  mutate(
    Strategy = factor(
      Strategy,
      levels = c("None", "Moderate", "Strong", "SDP")
    )
  )

# -----------------------------#
# 11) Summaries across stochastic replicates
# -----------------------------#

summary_time <- sim_all %>%
  group_by(Strategy, Time) %>%
  summarise(
    mean_cost = mean(CumCost),
    lo_cost = quantile(CumCost, 0.05),
    hi_cost = quantile(CumCost, 0.95),

    mean_occ = mean(OccupiedSites),
    lo_occ = quantile(OccupiedSites, 0.05),
    hi_occ = quantile(OccupiedSites, 0.95),

    mean_management = mean(CumManagementCost),
    mean_damage = mean(CumDamage),

    .groups = "drop"
  )

summary_final <- sim_all %>%
  filter(Time == Tmax) %>%
  group_by(Strategy) %>%
  summarise(
    mean_total_cost = mean(CumCost),
    sd_total_cost = sd(CumCost),
    q05_total_cost = quantile(CumCost, 0.05),
    q95_total_cost = quantile(CumCost, 0.95),

    mean_damage = mean(CumDamage),
    mean_management_cost = mean(CumManagementCost),

    mean_occupied_sites = mean(OccupiedSites),
    probability_eradicated = mean(OccupiedSites == 0),

    .groups = "drop"
  )

print(summary_final)

# -----------------------------#
# 12) SDP action frequencies
# -----------------------------#

sdp_actions <- sim_all %>%
  filter(Strategy == "SDP") %>%
  count(Time, Action) %>%
  group_by(Time) %>%
  mutate(Proportion = n / sum(n)) %>%
  ungroup() %>%
  mutate(
    Action = factor(
      Action,
      levels = action_names
    )
  )

# -----------------------------#
# 13) Figure panels
# -----------------------------#

# A. Local extinction probability under each action
df_ext <- tibble(
  Action = factor(action_names, levels = action_names),
  p_ext = as.numeric(p_ext)
)

p1 <- ggplot(
  df_ext,
  aes(x = Action, y = p_ext, group = 1)
) +
  geom_point(size = 2.5) +
  geom_line(linewidth = 0.7) +
  labs(
    x = "Management action",
    y = expression(p[ext])
  ) +
  coord_cartesian(ylim = c(0, 0.32)) +
  theme_minimal() +
  ggtitle("A.")
p1

# B. Colonisation probability as a function of occupied neighbours
df_col <- expand_grid(
  k = 0:5,
  Action = seq_len(n_actions)
) %>%
  mutate(
    beta_eff = beta_col * (1 - col_suppression[Action]),
    p_col = 1 - (1 - beta_eff)^k,
    Action = factor(
      action_names[Action],
      levels = action_names
    )
  )

p2 <- ggplot(
  df_col,
  aes(
    x = k,
    y = p_col,
    linetype = Action
  )
) +
  geom_line(linewidth = 0.9) +
  labs(
    x = "No. occupied neighbours k",
    y = expression(p[col](k)),
    linetype = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom"
  ) +
  ggtitle("B.")
p2

# C. Mean cumulative total cost, with 90% stochastic interval
# ------------------------------------------------------------
# Consistent colours across all worked examples
# ------------------------------------------------------------

cols <- c(
  None     = "grey80",
  Moderate = "grey60",
  Strong   = "grey35",
  SDP      = "#0072B2"
)

lts <- c(
  "None"     = "solid",
  "Moderate" = "dashed",
  "Strong"   = "dotdash",
  "SDP"      = "solid"
)

# ------------------------------------------------------------
# C. Mean cumulative total cost
# ------------------------------------------------------------

p_costs <-
  ggplot(
    summary_time,
    aes(
      x = Time,
      y = mean_cost
    )
  ) +
  
  # uncertainty bands
  geom_ribbon(
    aes(
      ymin = lo_cost,
      ymax = hi_cost,
      fill = Strategy
    ),
    alpha = 0.08,
    colour = NA
  ) +
  
  # Constant strategies
  geom_line(
    data = subset(summary_time, Strategy != "SDP"),
    aes(
      colour = Strategy,
    ),
    linewidth = 0.9
  ) +
  
  # SDP highlighted
  geom_line(
    data = subset(summary_time, Strategy == "SDP"),
    aes(
      colour = Strategy
    ),
    linewidth = 1.5
  ) +
  
  scale_colour_manual(values = cols) +
  scale_fill_manual(values = cols) +

  labs(
    x = "Time step",
    y = "Cumulative cost",
    colour = NULL,
    fill = NULL
  ) +
  
  guides(
    fill = "none"
  ) +
  
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  ggtitle("C.")

p_costs

# ------------------------------------------------------------
# D. Mean number of occupied sites through time
# ------------------------------------------------------------

p_occ <-
  ggplot(
    summary_time,
    aes(
      x = Time,
      y = mean_occ
    )
  ) +
  
  # uncertainty bands
  geom_ribbon(
    aes(
      ymin = lo_occ,
      ymax = hi_occ,
      fill = Strategy
    ),
    alpha = 0.08,
    colour = NA
  ) +
  
  # Constant strategies
  geom_line(
    data = subset(summary_time, Strategy != "SDP"),
    aes(
      colour = Strategy
    ),
    linewidth = 0.9
  ) +
  
  # SDP highlighted
  geom_line(
    data = subset(summary_time, Strategy == "SDP"),
    aes(
      colour = Strategy
    ),
    linewidth = 1.5
  ) +
  
  scale_colour_manual(values = cols) +
  scale_fill_manual(values = cols) +
  
  scale_y_continuous(
    breaks = 0:n_sites,
    limits = c(0, n_sites)
  ) +
  
  labs(
    x = "Time step",
    y = "Occupied sites",
    colour = NULL,
    fill = NULL
  ) +
  
  guides(
    fill = "none"
  ) +
  
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  
  ggtitle("D.")

p_occ

# ------------------------------------------------------------
# E. Decompose final expected cost into damage
#    and management expenditure
# ------------------------------------------------------------

df_cost_components <- summary_final %>%
  select(
    Strategy,
    Damage = mean_damage,
    `Management cost` = mean_management_cost
  ) %>%
  pivot_longer(
    cols = c(Damage, `Management cost`),
    names_to = "Component",
    values_to = "Cost"
  )

component_cols <- c(
  "Damage" = "grey35",
  "Management cost" = "grey75"
)

p_components <-
  ggplot(
    df_cost_components,
    aes(
      x = Strategy,
      y = Cost,
      fill = Component
    )
  ) +
  
  geom_col(
    width = 0.7
  ) +
  
  scale_fill_manual(
    values = component_cols
  ) +
  
  labs(
    x = NULL,
    y = paste0("Expected cumulative cost at T = ", Tmax),
    fill = NULL
  ) +
  
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  
  ggtitle("E.")

p_components

# ------------------------------------------------------------
# F. Which actions does the SDP actually use?
# ------------------------------------------------------------

action_cols <- c(
  "None"     = "grey75",
  "Moderate" = "#E69F00",
  "Strong"   = "#D55E00"
)

p_action <-
  ggplot(
    sdp_actions,
    aes(
      x = Time,
      y = Proportion,
      fill = Action
    )
  ) +
  
  geom_area(
    position = "stack",
    colour = NA
  ) +
  
  scale_fill_manual(
    values = action_cols
  ) +
  
  scale_y_continuous(
    labels = percent,
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0))
  ) +
  
  labs(
    x = "Time step",
    y = "SDP action frequency",
    fill = NULL
  ) +
  
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  
  ggtitle("F.")

p_action

# -----------------------------#
# 14) Optional policy plot
# -----------------------------#

# This panel shows how the SDP recommendation changes with invasion extent
# and remaining time. Because spatial configuration matters, multiple actions
# can be optimal for the same number of occupied sites; we therefore show the
# proportion of configurations assigned to each action.

# ------------------------------------------------------------
# Optional policy plot
# ------------------------------------------------------------

p_policy <-
  ggplot(
    df_policy_summary,
    aes(
      x = Time,
      y = OccupiedSites,
      fill = Action,
      alpha = Proportion
    )
  ) +
  
  geom_tile() +
  
  scale_fill_manual(
    values = action_cols
  ) +
  
  scale_y_continuous(
    breaks = 0:n_sites
  ) +
  
  scale_alpha_continuous(
    range = c(0.15, 1),
    guide = "none"
  ) +
  
  labs(
    x = "Time step",
    y = "Number of occupied sites",
    fill = NULL
  ) +
  
  theme_minimal() +
  theme(
    panel.grid = element_blank(),
    legend.position = "bottom"
  )

p_policy

# -----------------------------#
# 15) Assemble and save main figure
# -----------------------------#

final_plot <- (
  p1 + p2 + p_costs +
  p_occ + p_components + p_action
) +
  plot_layout(nrow = 2)

ggsave(
  "../SDPaper/figures/figure3.png",
  final_plot,
  dpi = 600,
  height = 6.5,
  width = 11
)

final_plot

