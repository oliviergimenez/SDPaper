#########################################################################
# Adaptive management via SDP and coypu - value iteration
# Olivier Gimenez, July 2025
#########################################################################

# SIS model with local logistic dynamics
# S = Susceptible (empty site) / I = Infecté (occupied site),
# Local logistic population growth if the site is occupied,
# Sites are cities where removal takes place (connected random graph)
# Monthly time steps, 5 years
# Higher extinction probability when population is low,
# Higher colonization probability if neighboring sites are occupied and abundant.
# At each time step, global decision (same action on all sites)
# Actions: no control, moderate control, or strong control
# Optimization via MDPtoolbox
# Cost = damage (abundance × damage cost) + control cost (depending on global action)

# Build a network of 8 sites with true 2022 abundance estimates from Gimenez et al. 2025
# Define a local logistic‐growth + harvest step and a simple extinction/colonisation rule
# Estimate transition matrices P via Monte-Carlo and rewards R from damage + control cost
# Solve the MDP with value iteration, then
# Forward‐simulate under the optimal policy

#----- required packages

library(MDPtoolbox)
library(igraph)
library(tidyverse)
library(patchwork)

library(leaflet)
library(tidygeocoder)



#---- initialisation

# network
villes <- c("Candillargues", "La Grande‑Motte", "Lansargues",
           "Marsillargues", "Mauguio", "Valergues",
           "Mudaison", "Saint‑Nazaire‑de‑Pézan")
pop_est <- c(80, 0, 125, 0, 50, 15, 0, 200) # estimated pop size in 2022
n_sites <- length(villes)
set.seed(666)
g <- sample_gnp(n_sites, p = 0.7)
#g <- sample_smallworld(1, n_sites, 3, 0.05)
#g <- sample_gnm(n_sites, 2 * n_sites)
while (!is_connected(g)) {
  g <- sample_gnp(n_sites, p = 0.7)
  #g <- sample_smallworld(1, n_sites, 2, 0.05)
  #g <- sample_gnm(n_sites, 2 * n_sites)
}
adj <- as_adjacency_matrix(g, sparse=FALSE)
plot(g)

# Parameters
Tmax <- 60 # (60 months)
K <- 250             # Capacité locale
r <- 0.7            # Taux de croissance
actions <- c(0, 0.1, 0.2)  # intensité harvest : rien / modéré / fort
control_cost <- c(0, 5, 10)
damage_cost <- 1
n_sims <- 100 # Monte-Carlo approx of the transition matrix

# States: 2^n_sites (each site is occupied or not)
state_list <- as.matrix(expand.grid(replicate(n_sites, c(0, 1), simplify=FALSE)))
n_states <- nrow(state_list)
n_actions <- length(actions)

#---- Model projection
simulate_next_state <- function(state, action, pop, adj) {
  next_state <- state
  for (i in 1:length(state)) {
    if (state[i] == 1) {
      # extinction prob is a function of N (the lower N, the higher extinction)
      p_ext <- exp(-0.15 * pop[i])
      if (runif(1) < p_ext) {
        next_state[i] <- 0
        pop[i] <- 0
      }
    } else {
      # colonization if neighbor occupied
      nei <- which(adj[i,] == 1)
      p_col <- 1 - prod(1 - 0.1 * state[nei])
      if (runif(1) < p_col) {
        next_state[i] <- 1
        pop[i] <- 1
      }
    }
  }
  list(state = next_state, pop = pop)
}

#----- Transition matrix and reward

# Initialisation
P <- vector("list", n_actions)
R <- matrix(0, n_states, n_actions)

for (a in 1:n_actions) {
  P[[a]] <- matrix(0, n_states, n_states)
  
  for (s in 1:n_states) {
    state <- state_list[s, ]
    pop <- ifelse(state == 1, pop_est, 0)
    # pop <- ifelse(state == 1, rpois(n_sites, lambda = 50), 0)
    
    # Local growth + harvest
    pop <- pop + r * pop * (1 - pop/K)
    pop <- pop - actions[a] * pop
    pop[pop < 0] <- 0
    
    # Simulate N simulations to approx transition matrix
    next_states <- matrix(0, nrow = n_sims, ncol = n_sites)
    for (sim in 1:n_sims) {
      res <- simulate_next_state(state, a, pop, adj)
      next_states[sim, ] <- res$state
    }
    next_states_df <- as.data.frame(next_states)
    state_ids <- apply(next_states_df, 1, function(row) {
      which(apply(state_list, 1, function(x) all(x == row)))
    })
    freqs <- table(state_ids)
    probs <- freqs / sum(freqs)
    P[[a]][s, as.numeric(names(probs))] <- probs
    
    # Reward = - (damages + control)
    R[s, a] <- - (damage_cost * sum(pop) + control_cost[a])
  }
}

# # Even though Monte-Carlo should cover every possible next state, 
# # it's good practice to force each row to sum to one:
# for(a in seq_len(n_actions)) {
#   row_sums <- rowSums(P[[a]])
#   P[[a]][row_sums > 0, ] <- P[[a]][row_sums > 0, ] / row_sums[row_sums > 0]
# }

#----- Value iteration and optimal policy
mdp_check(P, R)
#res <- mdp_value_iteration(P, R, discount = 0.9, epsilon = 0.01)
#res

horizon <- 60
res <- mdp_finite_horizon(P, R, discount = 0.99, horizon)
res

#----- Forward simulation under optimal policy -----------------------------

# Allocate memory for storage
pop_mat   <- matrix(NA, nrow = Tmax, ncol = n_sites)
state_mat <- matrix(NA, nrow = Tmax, ncol = n_sites)

# Initial condition
set.seed(123)
ville <- c("Candillargues", "La Grande‑Motte", "Lansargues",
          "Marsillargues", "Mauguio", "Valergues",
          "Mudaison", "Saint‑Nazaire‑de‑Pézan")
pop <- c(80, 0, 125, 0, 50, 15, 0, 200) # estimated pop size in 2022
state <- ifelse(pop > 0, 1, 0)

pop_mat[1,]   <- pop
state_mat[1,] <- state

# Step forward
for(t in 1:(Tmax-1)) {
  # Find index of current binary state in state_list
  sid <- which(apply(state_list, 1, function(x) all(x == state)))
  # Pick the global action from the optimal policy
  a   <- res$policy[sid]
  # Local logistic growth + global harvest
  pop <- ifelse(state == 1,
                pop + r * pop * (1 - pop / K),
                0)
  pop <- pop - actions[a] * pop
  pop[pop < 0] <- 0
  # Stochastic extinction / colonization
  out <- simulate_next_state(state, a, pop, adj)
  state <- out$state
  pop   <- out$pop
  # Record
  pop_mat[t+1,]   <- pop
  state_mat[t+1,] <- state
}



#----- Col/ext probabilities

# p_ext fonction de N
df_ext <- tibble(N = 0:100,
                 p_ext = exp(-0.15 * N))
# p_col fonction du nombre de voisins k
df_col <- tibble(k = 0:5,
                 p_col = 1 - (1 - 0.1)^k)

p1 <- ggplot(df_ext, aes(N, p_ext)) + 
  geom_line() +
  labs(title = "p_col(k)", x = "No. Occupied Neighbors k", y = "Colonisation Prob.") +
  theme_minimal()
p2 <- ggplot(df_col, aes(k, p_col)) + 
  geom_line() +
  labs(title = "p_ext(N)", x = "Local Abundance N", y = "Extinction Prob.") +
  theme_minimal()

p1 + p2


#----- Map of study area

# Our 8 cities
mes_villes <- c(
  "Candillargues", "La Grande-Motte", "Lansargues",
  "Marsillargues", "Mauguio", "Valergues",
  "Mudaison", "Saint-Nazaire-de-Pézan"
)

# OSM geocoding w/ estimated abundance
effectifs <- c(80, 0, 125, 0, 50, 15, 0, 200)

cities_df <- tibble(ville = mes_villes) %>%
  geocode(
    address      = ville,
    method       = "osm",
    lat          = latitude,
    long         = longitude,
    full_results = FALSE
  ) %>%
  mutate(ragondins = effectifs)

# Check
print(cities_df)
#> # A tibble: 8 × 4
#>   ville               latitude longitude ragondins
#>   <chr>                  <dbl>     <dbl>     <dbl>
#> 1 Candillargues          43.59     4.05         80
#> 2 La Grande-Motte        43.54     4.08          0
#> 3 Lansargues             43.66     4.03        125
#> 4 Marsillargues          43.66     4.18          0
#> 5 Mauguio                43.63     3.92         50
#> 6 Valergues              43.62     3.95         15
#> 7 Mudaison               43.66     3.96          0
#> 8 Saint-Nazaire-de-Pézan 43.58     4.15        200

# Build leaflet map
leaflet(cities_df) %>%
  # map background
  addProviderTiles(providers$OpenStreetMap,      group = "Plan") %>%
  addProviderTiles(providers$Esri.WorldImagery,  group = "Satellite") %>%
  # points w/ estimated abundance and popup info
  addCircleMarkers(
    lng         = ~longitude,
    lat         = ~latitude,
    #label       = ~ville,
    popup       = ~paste0(
      "<strong>", ville, "</strong><br/>",
      "Ragondins : ", ragondins
    ),
    #    radius      = 6,
    radius     = ~5 + sqrt(ragondins),
    fillOpacity = 0.3,
    color       = "red",
    group       = "Communes",
    # label = texte affiché, ici le nombre de ragondins
    label       = ~as.character(ragondins),
    labelOptions = labelOptions(
      noHide      = TRUE,      # toujours afficher
      direction   = "top",     # au-dessus du point
      textOnly    = TRUE,      # sans marqueur
      style       = list(
        "font-weight" = "bold",
        "font-size"   = "12px",
        "color"       = "black",
        "background"  = "rgba(255,255,255,0.7)",
        "padding"     = "2px"
      )
    ),
  ) %>%
  # Layers
  addLayersControl(
    baseGroups    = c("Plan", "Satellite"),
    overlayGroups = "Communes",
    options       = layersControlOptions(collapsed = FALSE)
  ) %>%
  # Legend
  addLegend(
    position = "bottomright",
    colors   = "red",
    labels   = "Commune",
    title    = "Points villes"
  ) %>%
  # Title
  addControl(
    html     = "<strong>Carte Plan – Effectifs de ragondins</strong>",
    position = "topright"
  )

#----- Coypu abundance dynamics per site (optimal policy)

# Assumes we have run the SDP-coypu-model.R script
# source("code/coypu/1-SDP-coypu-model.R")

# Build a tiny tibble of initial abundances by site
pop_init_df <- tibble(
  Site = mes_villes,
  init = pop_est
)

# Build abundance‐over‐time data frame
colnames(pop_mat) <- mes_villes
df_pop <- as_tibble(pop_mat) %>%
  mutate(Time = 1:Tmax) %>%
  pivot_longer(
    cols = -Time,
    names_to  = "Site",
    values_to = "Abundance"
  )

# Plot with a per‐facet dashed line at the initial pop
ggplot(df_pop, aes(Time, Abundance)) +
  geom_line() +
  facet_wrap(~ Site) + #, scales = "free_y") +
  #scale_y_continuous(limits = c(0, NA), expand = c(0, 0)) +
  geom_hline(
    data = pop_init_df,
    aes(yintercept = init),
    color     = "black",
    linetype  = "dashed") +
  labs(
    title = "Coypu abundance dynamics per site (optimal policy)",
    x     = "Time step",
    y     = "Abundance"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

#----- Network map colored by occupancy status

#see app.R

#----- Cumulative costs: same action all the time vs SDP optimal strategy

# Initial conditions
state_init <- ifelse(pop_est > 0, 1, 0)
pop_init   <- pop_est

# Simulator for a constant action index a ∈ {1,2,3}
simulate_const <- function(a) {
  pop   <- pop_init
  state <- state_init
  cost  <- numeric(Tmax)
  
  for (t in 1:Tmax) {
    # growth + harvest
    pop <- ifelse(state == 1,
                  pop + r * pop * (1 - pop / K),
                  0)
    pop <- pop - actions[a] * pop
    pop[pop < 0] <- 0
    
    # cost [damage + control]
    cost[t] <- sum(pop) * damage_cost + control_cost[a]
    
    # next state
    out     <- simulate_next_state(state, a, pop, adj)
    state   <- out$state
    pop     <- out$pop
  }
  cumsum(cost)
}

# Run for each constant strategy
cost_none     <- simulate_const(1)
cost_moderate <- simulate_const(2)
cost_strong   <- simulate_const(3)

# Calculate optimal actions & cost_step
action_opt <- apply(state_mat, 1, function(st) {
  sid <- which(apply(state_list, 1, function(x) all(x == st)))
  res$policy[sid]
})
cost_step_opt <- map_dbl(seq_len(Tmax), function(t) {
  sum(pop_mat[t, ]) * damage_cost + control_cost[action_opt[t]]
})
cum_cost_opt  <- cumsum(cost_step_opt)

# Combine all into one data.frame
df_costs <- tibble(
  Time     = rep(1:Tmax, 4),
  CumCost  = c(cost_none,
               cost_moderate,
               cost_strong,
               cum_cost_opt),
  Strategy = factor(
    rep(c("None", "Moderate", "Strong", "MDP‐Optimal"),
        each = Tmax),
    levels = c("None","Moderate","Strong","MDP‐Optimal")
  )
)

ggplot(df_costs, aes(x = Time, y = CumCost, color = Strategy)) +
  geom_line(size = 1.2) +
  scale_color_manual(
    values = c(
      "None"        = "grey50",
      "Moderate"    = "orange",
      "Strong"      = "red",
      "MDP‐Optimal" = "blue"
    )
  ) +
  labs(
    title    = "Cumulative Cost: Constant vs. MDP-Optimal Strategies",
    x        = "Time step",
    y        = "Cumulative Cost",
    color    = "Strategy"
  ) +
  theme_minimal() +
  theme(
    plot.title    = element_text(face = "bold", hjust = 0.5),
    legend.title  = element_text(face = "bold"),
    legend.position = "bottom"
  )




#----- Population dynamics according to optimal strategy and constant actions

# Define initial conditions & parameters
mes_villes <- c(
  "Candillargues", "La Grande-Motte", "Lansargues",
  "Marsillargues",  "Mauguio",         "Valergues",
  "Mudaison",       "Saint-Nazaire-de-Pézan"
)
pop_init  <- pop_est                                # vector length = n_sites
state_init<- ifelse(pop_init > 0, 1, 0)
Tmax      <- nrow(pop_mat)
actions   <- c(0, 0.1, 0.2)
damage_cost <- 1
control_cost<- c(0,5,10)

# Simulator that records pop × time for a single constant action a
simulate_mat <- function(a) {
  pop   <- pop_init
  state <- state_init
  M     <- matrix(0, nrow = Tmax, ncol = length(pop_init))
  
  for (t in 1:Tmax) {
    # 1) record
    M[t, ] <- pop
    
    # 2) growth + harvest
    pop <- ifelse(state == 1,
                  pop + r * pop * (1 - pop / K), 
                  0)
    pop <- pop - actions[a] * pop
    pop[pop < 0] <- 0
    
    # 3) next state & pop
    out   <- simulate_next_state(state, a, pop, adj)
    state <- out$state
    pop   <- out$pop
  }
  M
}

# Build the four matrices
mat_none     <- simulate_mat(1)
mat_moderate <- simulate_mat(2)
mat_strong   <- simulate_mat(3)
mat_opt      <- pop_mat   # from your MDP‐forward simulation

# Stack them into one data.frame
df_none <- as_tibble(mat_none) %>% 
  set_names(mes_villes) %>% 
  mutate(Time = 1:Tmax, Strategy = "None")

df_mod  <- as_tibble(mat_moderate) %>% 
  set_names(mes_villes) %>% 
  mutate(Time = 1:Tmax, Strategy = "Moderate")

df_str  <- as_tibble(mat_strong) %>% 
  set_names(mes_villes) %>% 
  mutate(Time = 1:Tmax, Strategy = "Strong")

df_opt  <- as_tibble(mat_opt) %>% 
  set_names(mes_villes) %>% 
  mutate(Time = 1:Tmax, Strategy = "SDP-Optimal")

# 4) Now bind them — all tibbles have exactly length(mes_villes)+2 columns
df_all <- bind_rows(df_none, df_mod, df_str, df_opt) %>% 
  pivot_longer(
    cols       = all_of(mes_villes),
    names_to   = "Site",
    values_to  = "Abundance"
  ) %>%
  mutate(
    Strategy = factor(Strategy, 
                      levels = c("None","Moderate","Strong","SDP-Optimal"))
  )

# Plot
ggplot(df_all, aes(Time, Abundance, color = Strategy, linetype = Strategy)) +
  geom_line(size = 1) +
  facet_wrap(~ Site, ncol = 4) +
  scale_color_manual(values = c(
    None         = "grey50",
    Moderate     = "orange",
    Strong       = "red",
    `SDP-Optimal`= "blue"
  )) +
  scale_linetype_manual(values = c(
    None         = "solid",
    Moderate     = "dashed",
    Strong       = "dotdash",
    `SDP-Optimal`= "twodash"
  )) +  
  labs(
    title    = "Abundance dynamics per site\n(Constant strategies vs. MDP-Optimal)",
    x        = "Time step",
    y        = "Abundance",
    color    = "Strategy"
  ) +
  theme_minimal() +
  theme(
    plot.title      = element_text(face = "bold", hjust = 0.5),
    legend.position = "bottom",
    strip.text      = element_text(face = "bold")
  )


#----- global optimal policy actions at each time

# Reconstruct the global optimal‐policy action at each time
action_seq <- tibble(
  Time     = 1:Tmax,
  Strategy = factor(action_opt,
                    levels = 1:3,
                    labels = c("None","Moderate","Strong"))
)

# Plot as a tile‐plot
ggplot(action_seq, aes(x = Time, y = 1, fill = Strategy)) +
  geom_tile() +
  scale_fill_manual(
    name   = "Action",
    values = c(None = "grey50", Moderate = "orange", Strong = "red")
  ) +
  scale_y_continuous(NULL, breaks = NULL) +   # drop the y‐axis
  scale_x_continuous(breaks = seq(1, Tmax, by = 5)) +
  labs(
    title = "Global management action over time",
    x     = "Time step"
  ) +
  theme_minimal() +
  theme(
    axis.title.y     = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    legend.position  = "bottom"
  )




#### Heatmap of the number of coypus to be removed

# Matrice du nombre prélevé : pop × intensité de prélèvement
harvest_mat <- pop_mat * actions[action_opt]

# Construire un df long
df_harvest <- as_tibble(harvest_mat) %>%
  set_names(mes_villes) %>%
  mutate(Time = 1:Tmax) %>%
  pivot_longer(cols = all_of(mes_villes),
               names_to = "Site",
               values_to = "Harvested")

ggplot(df_harvest, aes(x = Time, y = Site, fill = Harvested)) +
  geom_tile() +
  scale_fill_viridis_c(name = "No. removed") +
  labs(title = "Number of coypus removed (optimal policy)",
       x = "Time step", y = "Site") +
  theme_minimal()




## Probabilité de contenir la population (≤5 indiv.)
# Probabilité que chaque site ait ≤ 5 individus au fil du temps, 
# selon la stratégie.

df_summary <- df_all %>%
  group_by(Time, Strategy) %>%
  summarise(perc_low = mean(Abundance <= 5), .groups = "drop")

ggplot(df_summary, aes(Time, perc_low, color = Strategy)) +
  geom_line(size = 1.2) +
  scale_y_continuous(labels = scales::percent) +
  labs(title = "Proportion of sites with ≤ 5 coypus",
       x = "Time step", y = "Containment success",
       color = "Strategy") +
  theme_minimal()


# On part d’un état initial réel (tes estimations 2022), on applique 
# la politique optimale et on montre comment évolue la probabilité 
# d’extinction site par site au cours du temps.

# --- Exemple narratif pour le ragondin ---

# Etat initial (tes estimations 2022 déjà dans pop_est et state_init)
init_pop   <- pop_est
init_state <- ifelse(init_pop > 0, 1, 0)

# Re-simuler une trajectoire de 10 ans sous la politique optimale
Tshort <- 10
pop_traj   <- matrix(NA, nrow = Tshort, ncol = n_sites)
state_traj <- matrix(NA, nrow = Tshort, ncol = n_sites)

pop <- init_pop
state <- init_state
pop_traj[1,]   <- pop
state_traj[1,] <- state

for (t in 1:(Tshort-1)) {
  # Trouver l’état courant dans state_list
  sid <- which(apply(state_list, 1, function(x) all(x == state)))
  a   <- res$policy[sid]   # action optimale
  
  # Croissance + harvest
  pop <- ifelse(state == 1,
                pop + r * pop * (1 - pop/K),
                0)
  pop <- pop - actions[a] * pop
  pop[pop < 0] <- 0
  
  # Extinction / colonisation stochastique
  out   <- simulate_next_state(state, a, pop, adj)
  state <- out$state
  pop   <- out$pop
  
  # Stockage
  pop_traj[t+1,]   <- pop
  state_traj[t+1,] <- state
}

# Transformer en tibble long
df_traj <- as_tibble(pop_traj) %>%
  set_names(mes_villes) %>%
  mutate(Time = 1:Tshort) %>%
  pivot_longer(-Time, names_to="Site", values_to="Abundance") %>%
  mutate(p_ext = exp(-0.15 * Abundance))   # probabilité extinction locale

# ---- Graphique : probabilité extinction site × temps
ggplot(df_traj, aes(Time, p_ext, color=Site)) +
  geom_line(size=1.2) +
  geom_hline(yintercept=0.5, linetype="dashed", color="red") +
  labs(title="Extinction probability per site under optimal policy",
       subtitle="Initial state = estimated abundances (2022)",
       x="Time step (months)", y="Extinction probability") +
  theme_minimal() +
  theme(legend.position="bottom")
