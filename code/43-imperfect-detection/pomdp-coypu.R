###############################################################################
# Finite-horizon POMDP example (3 states x 3 actions) for an invasive species
#
# States:
#   1) "Eradicated"  = no animals detected (could still re-invade)
#   2) "Contained"   = invasion is contained (low abundance / low damage)
#   3) "Established" = invasion is established (high abundance / high damage)
#
# Actions:
#   1) "DoNothing" = do nothing (cheap, but risk of worsening)
#   2) "FertControl" = fertility control campaign (costly, but pushes system back to Contained)
#   3) "LethControl" = lethal control campaign (more costly, but changes system to Eradicated)
#
# Observations:
#   1) "Undetected"  = coypu not observed/detected
#   2) "Detected" = coypu observed/detected
#
# Rewards:
#   Reward = -(damage + management_cost)  (so higher = better)
#
###############################################################################
library(pomdp)
library(MDPtoolbox)
library(tidyverse)
library(patchwork)
library(scales)

set.seed(666)

states  <- c("Eradicated", "Contained", "Established"); S <- length(states)
actions <- c("DoNothing", "FertControl", "LethControl"); A <- length(actions)
observations = c("Undetected", "Detected")

#-----------------------------#
# 1) Transition matrices P[[a]]
#-----------------------------#
# P[[a]][s, s'] = Prob( next_state = s' | current_state = s, action = a )
# Rows sum to 1.

DoNothing = matrix(c(
    # to:   Eradicated Contained Established
    0.95,      0.05,     0.00,      # from Eradicated
    0.00,      0.80,     0.20,      # from Contained
    0.00,      0.00,     1.00       # from Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, states)
)

FertControl = matrix(c(
  # to:   Eradicated Contained Established
  0.98,      0.02,     0.00,      # from Eradicated
  0.15,      0.65,     0.20,      # from Contained
  0.05,      0.35,     0.60       # from Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, states)
)

LethControl = matrix(c(
    # to:   Eradicated Contained Established
    0.98,      0.02,     0.00,      # from Eradicated
    0.20,      0.75,     0.05,      # from Contained
    0.05,      0.75,     0.20       # from Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, states)
)

P_pomdp <- list(DoNothing, FertControl, LethControl)

# Quick check (optional): each row should sum to 1
lapply(P_pomdp, rowSums)

#-----------------------------#
# 2) Reward matrix R[s,a]
#-----------------------------#
# Here we use a toy "net benefit" style reward:
#   reward = -(damage + cost)
#
# Eradicated: no damage
# Contained: low damage
# Established: high damage
#
# Do nothing: no cost
# Fertility control: mid cost
# Lethal control: higher cost

damage <- c(Eradicated = 0, Contained = 4, Established = 14)
cost   <- c(DoNothing = 0, FertControl = 3, LethControl = 6)
reward <- outer(damage, cost, FUN = function(d, c) -(d + c))
rownames(reward) <- states
colnames(reward) <- actions

R_pomdp <- rbind(
  R_("DoNothing",   "Eradicated",  v = reward["Eradicated",  "DoNothing"]),
  R_("DoNothing",   "Contained",   v = reward["Contained",   "DoNothing"]),
  R_("DoNothing",   "Established", v = reward["Established", "DoNothing"]),
  R_("FertControl", "Eradicated",  v = reward["Eradicated",  "FertControl"]),
  R_("FertControl", "Contained",   v = reward["Contained",   "FertControl"]),
  R_("FertControl", "Established", v = reward["Established", "FertControl"]),
  R_("LethControl", "Eradicated",  v = reward["Eradicated",  "LethControl"]),
  R_("LethControl", "Contained",   v = reward["Contained",   "LethControl"]),
  R_("LethControl", "Established", v = reward["Established", "LethControl"])
)

#-----------------------------#
# 4) Detection probabilities O[s,a]
#-----------------------------#
# Here we assume that detectability is dependent on the state of the system and the action take.
# That is, we assume that 'DoNothing' reduces our ability to detect coypu, while both control measures
# have the same detection probabilities.
# Detection increases with the relative abundance of the coypu population. 

O_DoNothing <- matrix(c(
  # Undetected   Detected
  0.98, 0.02, # Detectability when Eradicated
  0.65, 0.35, # Detectability when Contained
  0.35, 0.65 # Detectability when Established
), nrow = S, byrow = TRUE, dimnames = list(states, observations))

O_FertControl <- matrix(c(
  # Undetected   Detected
  0.97, 0.03, # Detectability when Eradicated
  0.40, 0.60, # Detectability when Contained
  0.15, 0.85 # Detectability when Established
), nrow = S, byrow = TRUE, dimnames = list(states, observations))

O_LethControl <- matrix(c(
  # Undetected   Detected
  0.97, 0.03, # Detectability when Eradicated
  0.40, 0.60, # Detectability when Contained
  0.15, 0.85 # Detectability when Established
), nrow = S, byrow = TRUE, dimnames = list(states, observations))

O_pomdp <- list(
  DoNothing = O_DoNothing,
  FertControl = O_FertControl,
  LethControl = O_LethControl
)

#-----------------------------#
# 5) Solve POMDPs
#-----------------------------#
b0 <- c(Eradicated = 0, Contained = 0, Established = 1)
discount <- 0.95

# Specify the POMDP model for pomdp:
Coypu <- POMDP(
  states = states,
  actions = actions,
  observations = observations,
  transition_prob = P_pomdp,
  observation_prob = O_pomdp,
  reward = R_pomdp,
  discount = discount
)

# Solve the POMDP model:
sol_pomdp <- solve_POMDP(
  model = Coypu,
  horizon = NULL, # NULL used default of Inf 
  initial_belief = b0,
  method = "grid", # "grid", "enum", "twopass", "witness", or "incprune". The default is "grid"
)

# Inspect outputs:
# The first columns of the data.table returned by policy() provide the α-vectors coefficient (one per line), 
# the following column provides the optimal action
policy(sol_pomdp)

# Plot the policy graph:
plot_policy_graph(sol_pomdp, engine="visNetwork")

# dataviz
pol <- as.data.frame(policy(sol_pomdp))
alpha_mat <- as.matrix(pol[, seq_len(S)])
alpha_action <- as.character(pol[[S + 1]])

get_pomdp_action <- function(b) {
  vals <- as.numeric(alpha_mat %*% b)
  alpha_action[which.max(vals)]
}

get_pomdp_value <- function(b) {
  vals <- as.numeric(alpha_mat %*% b)
  max(vals)
}

update_belief <- function(b, action, observation) {
  a <- match(action, actions)
  o <- match(observation, observations)
  
  b_pred <- as.numeric(b %*% P_pomdp[[a]])
  likelihood <- O_pomdp[[a]][, o]
  b_new <- b_pred * likelihood
  
  if (sum(b_new) == 0) return(b_pred / sum(b_pred))
  b_new / sum(b_new)
}


# ---- Panel A: observation model ----

df_obs <- bind_rows(
  as.data.frame(O_DoNothing) %>% rownames_to_column("State") %>% mutate(Action = "DoNothing"),
  as.data.frame(O_FertControl) %>% rownames_to_column("State") %>% mutate(Action = "FertControl"),
  as.data.frame(O_LethControl) %>% rownames_to_column("State") %>% mutate(Action = "LethControl")
) %>%
  select(State, Action, Detected) %>%
  mutate(
    State = factor(State, levels = states),
    Action = factor(Action, levels = actions)
  )

action_cols <- c(
  "DoNothing" = "grey55",
  "FertControl" = "#E69F00",
  "LethControl" = "#D55E00"
)

p_obs <- ggplot(df_obs, aes(State, Detected, colour = Action, group = Action)) +
  geom_line(linewidth = 0.9) +
  geom_point(size = 2.2) +
  scale_colour_manual(values = action_cols) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  labs(
    x = "True invasion state",
    y = "Probability of observing coypu",
    colour = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  ggtitle("A.")
p_obs

# ---- Panel B: optimal policy over the belief simplex ----

simplex_xy <- function(b) {
  tibble(
    x = b[2] + 0.5 * b[3],
    y = (sqrt(3) / 2) * b[3]
  )
}

grid_step <- 0.02

belief_grid <- expand_grid(
  b_Eradicated = seq(0, 1, by = grid_step),
  b_Contained = seq(0, 1, by = grid_step)
) %>%
  mutate(b_Established = 1 - b_Eradicated - b_Contained) %>%
  filter(b_Established >= -1e-10) %>%
  mutate(b_Established = pmax(0, b_Established)) %>%
  rowwise() %>%
  mutate(
    Action = get_pomdp_action(c(b_Eradicated, b_Contained, b_Established)),
    x = b_Contained + 0.5 * b_Established,
    y = (sqrt(3) / 2) * b_Established
  ) %>%
  ungroup() %>%
  mutate(Action = factor(Action, levels = actions))

triangle <- tibble(
  x = c(0, 1, 0.5, 0),
  y = c(0, 0, sqrt(3) / 2, 0)
)

b_example <- c(0.10, 0.15, 0.75)
xy_example <- simplex_xy(b_example)

p_simplex <- ggplot(belief_grid, aes(x, y, colour = Action)) +
  geom_point(size = 2.0, alpha = 0.90) +
  geom_path(
    data = triangle,
    aes(x, y),
    inherit.aes = FALSE,
    linewidth = 0.8
  ) +
  geom_point(
    data = xy_example,
    aes(x, y),
    inherit.aes = FALSE,
    shape = 21,
    fill = "white",
    size = 3.2,
    stroke = 1
  ) +
  annotate("text", x = -0.02, y = -0.035, label = "Eradicated", hjust = 0, size = 3.2) +
  annotate("text", x = 1.02, y = -0.035, label = "Contained", hjust = 1, size = 3.2) +
  annotate("text", x = 0.5, y = sqrt(3)/2 + 0.035, label = "Established", hjust = 0.5, size = 3.2) +
  scale_colour_manual(values = action_cols) +
  coord_equal(
    xlim = c(-0.04, 1.04),
    ylim = c(-0.06, 0.93),
    clip = "off"
  ) +
  labs(x = NULL, y = NULL, colour = NULL) +
  theme_void() +
  theme(
    legend.position = "bottom",
    plot.margin = margin(10, 10, 10, 10)
  ) +
  ggtitle("B.")
p_simplex

# ---- Panel C: example belief update ----

b_start <- c(Eradicated = 0.10, Contained = 0.30, Established = 0.60)
a_start <- get_pomdp_action(b_start)

b_undetected <- update_belief(b_start, a_start, "Undetected")
b_detected <- update_belief(b_start, a_start, "Detected")

df_update <- bind_rows(
  tibble(Belief = "Prior", State = states, Probability = b_start),
  tibble(Belief = "After Undetected", State = states, Probability = b_undetected),
  tibble(Belief = "After Detected", State = states, Probability = b_detected)
) %>%
  mutate(
    Belief = factor(Belief, levels = c("Prior", "After Undetected", "After Detected")),
    State = factor(State, levels = states)
  )

state_cols <- c(
  "Eradicated" = "grey70",
  "Contained" = "#E69F00",
  "Established" = "#D55E00"
)

p_update <- ggplot(df_update, aes(Belief, Probability, fill = State)) +
  geom_col(width = 0.68) +
  scale_fill_manual(values = state_cols) +
  scale_y_continuous(
    labels = percent,
    limits = c(0, 1),
    expand = expansion(mult = c(0, 0.02))
  ) +
  labs(
    x = NULL,
    y = "Belief probability",
    fill = NULL,
    subtitle = paste0("Action taken: ", a_start)
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  ggtitle("C.")
p_update

# ---- Perfect-information MDP benchmark ----

mdp_sol <- mdp_value_iteration(
  P = P_pomdp,
  R = reward,
  discount = discount
)

perfect_action <- function(true_state) {
  actions[mdp_sol$policy[true_state]]
}

# ---- Simulation engine ----

draw_next_state <- function(state_id, action) {
  a <- match(action, actions)
  sample(seq_len(S), size = 1, prob = P_pomdp[[a]][state_id, ])
}

draw_observation <- function(next_state_id, action) {
  a <- match(action, actions)
  sample(observations, size = 1, prob = O_pomdp[[a]][next_state_id, ])
}

simulate_policy <- function(
    strategy = c("Naive", "POMDP", "Perfect information"),
    Tmax = 40,
    seed = NULL
) {
  strategy <- match.arg(strategy)
  if (!is.null(seed)) set.seed(seed)
  
  true_state <- match("Established", states)
  belief <- b0
  last_observation <- "Detected"
  
  out <- vector("list", Tmax)
  
  for (t in seq_len(Tmax)) {
    
    if (strategy == "POMDP") {
      action <- get_pomdp_action(belief)
    } else if (strategy == "Perfect information") {
      action <- perfect_action(true_state)
    } else {
      action <- ifelse(last_observation == "Undetected", "DoNothing", "LethControl")
    }
    
    step_cost <- -reward[true_state, action]
    
    next_state <- draw_next_state(true_state, action)
    obs <- draw_observation(next_state, action)
    
    if (strategy == "POMDP") {
      belief <- update_belief(belief, action, obs)
    }
    
    last_observation <- obs
    
    out[[t]] <- tibble(
      Time = t,
      Strategy = strategy,
      TrueState = states[true_state],
      Action = action,
      Observation = obs,
      Cost = step_cost
    )
    
    true_state <- next_state
  }
  
  bind_rows(out) %>% mutate(CumCost = cumsum(Cost))
}

# ---- Panel D: repeated simulation comparison ----

Tmax_sim <- 40
n_rep <- 1000
strategies <- c("Naive", "POMDP", "Perfect information")

sim_compare <- map_dfr(
  seq_len(n_rep),
  function(rep_id) {
    map_dfr(
      strategies,
      function(strategy) {
        simulate_policy(
          strategy = strategy,
          Tmax = Tmax_sim,
          seed = 20000 + rep_id
        ) %>%
          mutate(Replicate = rep_id)
      }
    )
  }
)

summary_compare <- sim_compare %>%
  group_by(Strategy, Time) %>%
  summarise(
    mean_cost = mean(CumCost),
    lo_cost = quantile(CumCost, 0.05),
    hi_cost = quantile(CumCost, 0.95),
    .groups = "drop"
  ) %>%
  mutate(
    Strategy = factor(
      Strategy,
      levels = c("Naive", "Perfect information", "POMDP")
    )
  )

strategy_cols <- c(
  "Naive" = "grey65",
  "Perfect information" = "grey20",
  "POMDP" = "#0072B2"
)

p_cost <- ggplot(summary_compare, aes(Time, mean_cost)) +
  geom_ribbon(
    aes(ymin = lo_cost, ymax = hi_cost, fill = Strategy),
    alpha = 0.08,
    colour = NA
  ) +
  geom_line(
    data = subset(summary_compare, Strategy != "POMDP"),
    aes(colour = Strategy),
    linewidth = 0.9
  ) +
  geom_line(
    data = subset(summary_compare, Strategy == "POMDP"),
    aes(colour = Strategy),
    linewidth = 1.5
  ) +
  scale_colour_manual(values = strategy_cols) +
  scale_fill_manual(values = strategy_cols) +
  guides(fill = "none") +
  labs(
    x = "Time step",
    y = "Cumulative cost",
    colour = NULL
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    panel.grid.minor = element_blank()
  ) +
  ggtitle("D.")
p_cost

# ---- Assemble ----

figure_pomdp <- (
  p_obs + p_simplex +
    p_update + p_cost
) +
  plot_layout(nrow = 2)

figure_pomdp

ggsave(
  "../SDPaper/figures/figure4.png",
  figure_pomdp,
  dpi = 600,
  width = 9,
  height = 8
)

#---- Useful outputs for the text

b_example
# [1] 0.10 0.15 0.75

get_pomdp_action(b_example)
#[1] "LethControl"

get_pomdp_value(b_example)
#[1] -96.03587

b_start
#Eradicated   Contained Established 
#0.1         0.3         0.6 

a_start
#[1] "LethControl"

b_undetected
#Eradicated   Contained Established 
#0.38520521  0.57202002  0.04277476 

b_detected
#Eradicated   Contained Established 
#0.01071042  0.77137811  0.21791147

final_costs <- sim_compare %>%
  filter(Time == Tmax_sim) %>%
  group_by(Strategy) %>%
  summarise(
    mean_final_cost = mean(CumCost),
    q05 = quantile(CumCost, 0.05),
    q95 = quantile(CumCost, 0.95),
    .groups = "drop"
  )

print(final_costs)

## A tibble: 3 × 4
#Strategy            mean_final_cost   q05   q95
#<chr>                         <dbl> <dbl> <dbl>
#  1 Naive                          192.  54    352.
#  2 POMDP                          175.  60.0  322 
#  3 Perfect information            144.  40    280.

