#########################################################################
# Adaptive management via SDP and wolf - value iteration
# adapted from code by Lucile Marescot and Guillaume Chapron
# Olivier Gimenez, July 2025
#########################################################################

library(MDPtoolbox)
library(tidyverse)

#----- Demographic and management parameters

# Maximum number of individuals in packs and as dispersers
N_pack_max <- 400
N_disp_max <- 100

# Carrying capacity (total population limit)
K <- 550

# Discretized state space for number of individuals in packs and dispersers
seq_pack <- seq(0, N_pack_max, 5)
seq_disp <- seq(0, N_disp_max, 5)

# Range of management actions (e.g., harvest intensity), from 0 to 1
seq_H <- seq(0, 1, 0.01)
stepind <- 10 # Used in state discretization/interpolation

# Reward function parameters
lambda_min <- 1     # minimum acceptable population growth rate
lambda_max <- 1.06  # maximum acceptable growth rate
coeff <- -1600      # penalty coefficient used in reward calculation
meute_min <- 20     # minimum number of packs required for viability

#----- State space 

# Number of possible states (all combinations of pack and disperser levels)
nb_states <- length(seq_pack) * length(seq_disp)

# Matrix storing the state index and associated (pack, disperser) values
# Columns: [state_id, number_in_pack, number_of_dispersers]
state_index <- matrix(0, nrow = nb_states, ncol = 3)

# Fill in the state index matrix
col2 <- NULL
for (i in 1:length(seq_pack)) {
  col2 <- c(col2, rep(seq_pack[i], length(seq_disp)))
}

state_index[,1] <- 1:nb_states
state_index[,2] <- col2
#state_index[,2] <- rep(seq_pack, each = length(seq_disp))
state_index[,3] <- rep(seq_disp, length(seq_pack))

#----- Biological parameters 

# Average pack size used in calculations (not directly used in this script, but informative)
pack_size <- 4  
# Average litter size per reproducing female per year
litter_size <- 3.5  
# Fecundity rate (offspring per alpha): assume only alphas reproduce, and only females (÷2)
f <- litter_size / 2  # = 1.75 pups per alpha female per year
# Annual survival probability (assumed identical for all stages here)
phi <- 0.84  # Can be adjusted or made stage-specific
# Probability that a disperser successfully establishes a new pack
pestab <- 0.6  
# Base dispersal probability (used in the density-dependent dispersal formula)
pdisp <- 0.4

#----- Demographic model (1-year projection)

# Function that projects the population one year ahead under a given action h
dynamic <- function (pack_pop, disp_pop, phi, p_es, p_di, f, h, K) {

  Stagenames <- c("juvenile","dispersers","subordinate", "alpha")
  NStages <- 4
  
  # Leslie-type projection matrix M (stage-structured)  
  M <- array(data = 0, dim = c(NStages, NStages)) 
  rownames(M) <- colnames(M) <- Stagenames
  
  # Building the projection matrix
  phi_j <- phi; phi_s <- phi; phi_a <- phi; phi_d <- phi
  M[1,4] <- f * phi_a               # alpha wolves produce juveniles
  M[2,1] <- phi_j * p_di            # juveniles disperse
  M[2,2] <- phi_d * (1 - p_es)      # dispersers that do not establish
  M[2,3] <- phi_s * p_di            # subordinates disperse
  M[3,1] <- phi_j * (1 - p_di)      # juveniles become subordinates
  M[3,3] <- phi_s * (1 - p_di)      # subordinates remain
  M[4,2] <- phi_d * p_es            # dispersers that establish become alphas
  M[4,4] <- phi_a                   # alphas survive
  
  # Stable stage distribution (eigenvector of M)
  rightV <- eigen(M)
  stableStage <- Re(rightV$vectors[,1])
  stableStage <- stableStage / sum(stableStage)
  
  # Use stable proportions to allocate individuals among stages  
  N_j <- pack_pop * stableStage[1] / sum(stableStage[c(1,3,4)])
  N_s <- pack_pop * stableStage[3] / sum(stableStage[c(1,3,4)])
  N_a <- pack_pop * stableStage[4] / sum(stableStage[c(1,3,4)])
  N_d <- disp_pop
  
  # Apply management action to individuals in packs
  popvector <- c(N_j, N_d, N_s, N_a)
  popvector[1] <- popvector[1]*(1-h)
  popvector[3] <- popvector[3]*(1-h)
  popvector[4] <- popvector[4]*(1-h)
  
  # Logistic growth (Miller et al. 2002)
  Mat <- M - diag(4)
  tot <- sum(popvector)
  if (tot > K) tot <- K
  next_pop <- popvector + ((K - tot) / K) * (Mat %*% popvector)
  
  # Outputs: pack pop, dispersers, total, pack number, average pack size
  nextpack_pop <- sum(next_pop[c(1,3,4)])
  nextNmeute <- next_pop[4] / 2
  return(c(nextpack_pop, 
           next_pop[2], 
           nextpack_pop + next_pop[2], 
           nextNmeute))
}

#----- Monte Carlo simulations: Building transition and reward matrices

# Initialize transition matrices (one per action)
transition <- array(0, dim = c(nb_states, nb_states, length(seq_H)))

# Reward matrix: rows = states, columns = actions
utility <- matrix(0, nrow = nb_states, ncol = length(seq_H))

#----- Loop over all states and actions

for (k in 1:nb_states) {
  
  # Extract the number of individuals in packs and dispersers for state k
  i <- state_index[k,2]
  j <- state_index[k,3]
  
  # Try all actions
  for (h in 1:length(seq_H)) {
    
    # Project population one year ahead under action h
    next_pop <- dynamic(i, j, phi, pestab, pdisp, f, seq_H[h], K)
    
    # # Calculate population growth rate (PGR), with safeguard for 0 denominator
    # pgr <- if ((i + j) > 0) next_pop[3] / (i + j) else 0
    # 
    # # Assign reward only if PGR is within acceptable range and enough packs
    # if (!is.na(pgr) && !is.na(out[4]) && pgr >= lambda_min && pgr <= lambda_max && out[4] >= meute_min) {
    #   utility[k, h] <- coeff * (pgr - lambda_min) * (pgr - lambda_max)  # parabolic reward
    # } else {
    #   utility[k, h] <- 0  # No reward (or penalty) otherwise
    # }
    ifelse((i+j) > 0, 
           pgr <- next_pop[3] / (i+j), 
           pgr <- 0)
    ifelse ((pgr >= lambda_min & pgr <= lambda_max & next_pop[4] >= meute_min),   
            utility[k,h] <- coeff * (pgr-lambda_min) * (pgr-lambda_max), 
            utility[k,h] <- 0)
    
    # # Cap population values to avoid exceeding state space boundaries
    # npack <- min(out[1], N_pack_max)
    # ndisp <- min(out[2], N_disp_max)
    # 
    # # Find nearest lower and upper grid points for interpolation
    # top_pack <- ceiling(npack / stepind) * stepind
    # bot_pack <- floor(npack / stepind) * stepind
    # top_disp <- ceiling(ndisp / stepind) * stepind
    # bot_disp <- floor(ndisp / stepind) * stepind
    # 
    # # Compute interpolation weights (linear) for pack and disperser dimensions    
    # wp <- if (!is.na(top_pack) && !is.na(bot_pack) && top_pack != bot_pack) {
    #   (npack - bot_pack) / (top_pack - bot_pack)
    # } else {
    #   1
    # }
    # 
    # wd <- if (!is.na(top_disp) && !is.na(bot_disp) && top_disp != bot_disp) {
    #   (ndisp - bot_disp) / (top_disp - bot_disp)
    # } else {
    #   1
    # }
    # 
    # # Helper to retrieve the row index of a given (pack, disperser) state
    # states <- function(pack, disp) which(state_index[, 2] == pack & state_index[, 3] == disp)
    # 
    # # Bilinear interpolation: assign transition probabilities to 4 surrounding states
    # transition[[h]][k, states(top_pack, top_disp)] <- wp * wd
    # transition[[h]][k, states(bot_pack, top_disp)] <- (1 - wp) * wd
    # transition[[h]][k, states(bot_pack, bot_disp)] <- (1 - wp) * (1 - wd)
    # transition[[h]][k, states(top_pack, bot_disp)] <- wp * (1 - wd)

    if(next_pop[1] > N_pack_max) next_pop[1]<-N_pack_max
    if(next_pop[2] > N_disp_max) next_pop[2]<-N_disp_max
    
    top_pack <- ceiling(next_pop[1]/stepind)*stepind
    down_pack <- floor(next_pop[1]/stepind)*stepind
    top_disp <- ceiling(next_pop[2]/stepind)*stepind
    down_disp <- floor(next_pop[2]/stepind)*stepind
    
    ifelse(top_pack != down_pack, interpolation_sup <- (next_pop[1]-down_pack)/(top_pack- down_pack), interpolation_sup<-1)
    ifelse(top_pack != down_pack, interpolation_inf <- (top_pack-next_pop[1])/(top_pack- down_pack), interpolation_inf<-1)
    
    ifelse(top_disp != down_disp, interpol_dispsup <- (next_pop[2] - down_disp)/(top_disp - down_disp),  interpol_dispsup <- 1)
    ifelse(top_disp != down_disp, interpol_dispinf <- (top_disp - next_pop[2])/(top_disp- down_disp),  interpol_dispinf <- 1)
    
    next_psup_dispsup <- which(state_index[,3] == top_disp & state_index[,2] == top_pack)
    
    transition[k,next_psup_dispsup,h]<-(interpolation_sup * interpol_dispsup)
    next_pinf_dispsup = which(state_index[,3] == top_disp & state_index[,2] == down_pack)
    transition[k,next_pinf_dispsup,h]<-(interpolation_inf * interpol_dispsup)
    next_pinf_dispinf = which(state_index[,3] == down_disp & state_index[,2] == down_pack)
    transition[k,next_pinf_dispinf,h]<-(interpolation_inf * interpol_dispinf)
    next_psup_dispinf = which(state_index[,3] == down_disp & state_index[,2] == top_pack)
    transition[k,next_psup_dispinf,h]<-(interpolation_sup * interpol_dispinf)
  }
}

# 
# # Ensure state 1 (often absorbing or dummy) has a valid row-sum of 1 in each matrix
# # Normalize each row in transition matrices so that rows sum to 1
# for (h in seq_along(transition)) {
#   for (k in 1:nrow(transition[[h]])) {
#     row_sum <- sum(transition[[h]][k, ])
#     if (row_sum > 0 && !is.na(row_sum)) {
#       transition[[h]][k, ] <- transition[[h]][k, ] / row_sum
#     } else {
#       # fallback if row has all zeros (e.g. absorbing state): stay in the same state
#       transition[[h]][k, ] <- 0
#       transition[[h]][k, k] <- 1
#     }
#   }
# }


#----- Value Iteration (MDP Solution)

# Run value iteration algorithm
P <- transition  # transition probabilities
R <- utility     # rewards
mdp_check(P, R)  # sanity check
S <- mdp_value_iteration(P, R, discount = 0.99, epsilon = 0.001)

horizon <- 10
S <- mdp_finite_horizon(P, R, discount = 0.99, horizon)

#----- 1. dataviz optimal policy

# Build df w/ optimal policy
policy_df <- data.frame(
  pack = state_index[,2],
  disp = state_index[,3],
  h = seq_H[S$policy]
)

# Delete rows w/ avec NA
policy_df <- na.omit(policy_df)

# Optimal management policy (value iteration)
ggplot(policy_df, aes(x = pack, y = disp, fill = h)) +
  geom_tile() +
  scale_fill_viridis_c(name = "Harvest intensity (h)", option = "D") +
  labs(
    x = "Number of individuals in packs",
    y = "Number of dispersers",
    title = "Optimal management policy (value iteration)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5),
    legend.position = "right"
  )


#----- 2. dataviz convergence 

# gammas <- c(0.7, 0.8, 0.9, 0.95, 0.99)
# values_df <- data.frame()
# 
# for (g in gammas) {
#   sol <- mdp_value_iteration(P, R, discount = g, epsilon = 0.001)
#   values_df <- rbind(values_df, data.frame(
#     gamma = g,
#     state = 1:nb_states,
#     value = sol$V
#   ))
# }
# 
# ggplot(values_df, aes(x = state, y = value, color = as.factor(gamma))) +
#   geom_line() +
#   labs(
#     x = "State index",
#     y = "Value function",
#     color = "Discount factor",
#     title = "Value function by discount factor (γ)"
#   ) +
#   theme_minimal()

#----- dataviz growth range in check

simulate_trajectory <- function(state_index, policy, seq_H, dynamic_fun, 
                                init_pack, init_disp, n_years, phi, pestab, pdisp, f, K) {
  pack <- numeric(n_years)
  disp <- numeric(n_years)
  total <- numeric(n_years)
  nmeute <- numeric(n_years)
  pgr <- numeric(n_years)
  h_vec <- numeric(n_years)
  
  pack[1] <- init_pack
  disp[1] <- init_disp
  total[1] <- init_pack + init_disp
  nmeute[1] <- init_pack / pack_size  # approx
  
  for (t in 1:(n_years - 1)) {
    # Find closest state index in the state space
    idx <- which.min((state_index[,2] - pack[t])^2 + (state_index[,3] - disp[t])^2)
    
    # Apply optimal action
    h_idx <- policy[idx]
    h <- seq_H[h_idx]
    h_vec[t] <- h
    
    # Simulate next state
    out <- dynamic_fun(pack[t], disp[t], phi, pestab, pdisp, f, h, K)
    
    pack[t+1] <- out[1]
    disp[t+1] <- out[2]
    total[t+1] <- out[3]
    nmeute[t+1] <- out[4]
    
    # Compute PGR
    pgr[t+1] <- if ((pack[t] + disp[t]) > 0) total[t+1] / total[t] else NA
  }
  
  return(data.frame(
    year = 1:n_years,
    pack = pack,
    disp = disp,
    total = total,
    pgr = pgr,
    nmeute = nmeute,
    h = h_vec
  ))
}


trajectory <- simulate_trajectory(
  state_index = state_index,
  policy = S$policy,
  seq_H = seq_H,
  dynamic_fun = dynamic,
  init_pack = 100,     # initial number of individuals in packs
  init_disp = 20,      # initial number of dispersers
  n_years = 30,
  phi = phi,
  pestab = pestab,
  pdisp = pdisp,
  f = f,
  K = K
)

ggplot(trajectory, aes(x = year, y = pgr)) +
  geom_line(color = "steelblue", linewidth = 1.2) +
  geom_hline(yintercept = lambda_min, linetype = "dashed", color = "red") +
  geom_hline(yintercept = lambda_max, linetype = "dashed", color = "red") +
  labs(
    x = "Year",
    y = "Population Growth Rate (PGR)",
    title = "PGR over time with target bounds"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5)
  )


# Reshape to long format for ggplot
trajectory_long <- trajectory %>%
  select(year, pack, disp, total) %>%
  pivot_longer(cols = c(pack, disp, total), 
               names_to = "component", 
               values_to = "abundance")

# Plot with color legend
ggplot(trajectory_long, aes(x = year, y = abundance, color = component, linetype = component)) +
  geom_line(size = 1.2) +
  scale_color_manual(
    values = c("pack" = "forestgreen", "disp" = "orange", "total" = "black"),
    labels = c("Pack individuals", "Dispersers", "Total population")
  ) +
  scale_linetype_manual(
    values = c("pack" = "dashed", "disp" = "dotted", "total" = "solid"),
    labels = c("Pack individuals", "Dispersers", "Total population")
  ) +
  labs(
    x = "Year",
    y = "Abundance",
    color = "Population component",
    linetype = "Population component",
    title = "Population size over time by component"
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    legend.title = element_text(face = "bold"),
    plot.title = element_text(face = "bold", hjust = 0.5)
  )


# dataviz management effort over time

ggplot(trajectory, aes(x = year, y = h)) +
  geom_step(color = "firebrick", size = 1.2) +
  labs(
    x = "Year",
    y = "Harvest intensity (h)",
    title = "Management action over time"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

########----------------- in practice ------------------####################

# « que fait un gestionnaire s’il a ce nombre de loups en meute et de 
# dispersants ? »

#-- option 1 

# on fixe une valeur de croissance souhaitée, par exemple λ = 1.05.
# on identifie dans la matrice de politique optimale les couples (pack, dispersants) qui permettent de rester dans cette zone.
# en pratique, on filtre policy_df pour les états où le PGR projeté est ≈ 1.05.
# ensuite on surimpose une zone colorée (patate de Christophe) sur le graphique pack vs disp 
# qui montre l’intervalle d’actions donnant λ ≈ 1.05.

# matérialise la gamme de possibles : 
# si le gestionnaire veut λ = 1.05, voici les intensités de prélèvement 
# qu’il peut appliquer selon l’état de la population.

# On calcule le taux de croissance projeté (pgr) sous la politique optimale, 
# puis on met en évidence les états où ce taux est proche de la cible.

# 1. Ajouter PGR projeté sous la politique optimale
policy_df$pgr <- NA

for (k in 1:nrow(policy_df)) {
  i <- policy_df$pack[k]
  j <- policy_df$disp[k]
  h <- policy_df$h[k]
  out <- dynamic(i, j, phi, pestab, pdisp, f, h, K)
  if ((i + j) > 0) {
    policy_df$pgr[k] <- out[3] / (i + j)
  }
}

# 2. Définir un intervalle autour de 1.05 (tolérance ±0.01)
target_lambda <- 1.05
tol <- 0.01

policy_df$lambda_target <- abs(policy_df$pgr - target_lambda) < tol

# 3. Visualiser
ggplot(policy_df, aes(x = pack, y = disp)) +
  geom_tile(aes(fill = h)) +
  geom_point(data = subset(policy_df, lambda_target == TRUE),
             aes(x = pack, y = disp), 
             color = "red", size = 2, alpha = 0.7) +
  scale_fill_viridis_c(name = "Harvest intensity (h)") +
  labs(
    title = expression("States leading to " ~ lambda %approx% 1.05),
    x = "Pack individuals", y = "Dispersers"
  ) +
  theme_minimal()


#-- option 2


# on sait déjà quel h est optimal pour chaque état (dans policy_df$h).
# on peut convertir h en nombre d’individus prélevés :
  
# Traduction en « nombre d’individus prélevés ». 
# On convertit h en nombre d’individus retirés de la population.

# on montre directement : « Si aujourd’hui j’ai 100 
# loups en meute et 75 dispersants, la politique optimale est de 
# prélever environ 15 individus. »

# Nombre d’individus prélevés
policy_df$harvest_number <- round(policy_df$h * (policy_df$pack + policy_df$disp))

ggplot(policy_df, aes(x = pack, y = disp, fill = harvest_number)) +
  geom_tile() +
  scale_fill_viridis_c(name = "Individuals harvested", option = "C") +
  labs(
    title = "Optimal harvest in number of wolves",
    x = "Pack individuals", y = "Dispersers"
  ) +
  theme_minimal()

# Choisir l'état cible
target_pack <- 100
target_disp <- 75

# Trouver l'index de l'état le plus proche dans ta grille
idx <- which.min((state_index[,2] - target_pack)^2 + (state_index[,3] - target_disp)^2)

# Récupérer l'action optimale (indice) et convertir en valeur de h
h_idx <- S$policy[idx]
h_opt <- seq_H[h_idx]

# Nombre d'individus prélevés
N_total <- state_index[idx,2] + state_index[idx,3]
harvest_number <- round(h_opt * N_total)

cat("État le plus proche : packs =", state_index[idx,2], 
    " dispersers =", state_index[idx,3], "\n")
cat("Action optimale (h) =", h_opt, "\n")
cat("Nombre d'individus prélevés ≈", harvest_number, "\n")


#-- option 3

# dans option 2
# on prend une situation initiale 100 meutes, 75 dispersants 
# on montre que la politique optimale recommande h = 0.07, soit 12 loups. 
# now on simule la trajectoire sur 10 ans en montrant que λ reste dans la cible.

# --- Exemple état initial
init_pack <- 100
init_disp <- 75

# Simulation avec ta fonction déjà définie
traj_ex <- simulate_trajectory(
  state_index = state_index,
  policy = S$policy,
  seq_H = seq_H,
  dynamic_fun = dynamic,
  init_pack = init_pack,
  init_disp = init_disp,
  n_years = 10,
  phi = phi,
  pestab = pestab,
  pdisp = pdisp,
  f = f,
  K = K
)

# Action optimale et nombre prélevé à t=1
h_init <- traj_ex$h[1]
harvest_init <- round(h_init * (init_pack + init_disp))

# Graphe sur le taux de croissance (λ)
ggplot(traj_ex, aes(x = year, y = pgr)) +
  geom_line(color = "black", size = 1.2) +
  geom_point(color = "blue", size = 2) +
  geom_hline(yintercept = lambda_min, linetype = "dashed", color = "red") +
  geom_hline(yintercept = lambda_max, linetype = "dashed", color = "red") +
  labs(
    title = paste0("Initial state: ", init_pack, " in packs, ", init_disp, 
                   " dispersers → h = ", round(h_init, 2), 
                   " (", harvest_init, " wolves)"),
    x = "Year", 
    y = expression("Population growth rate (λ)")
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(face = "bold", hjust = 0.5)
  )

