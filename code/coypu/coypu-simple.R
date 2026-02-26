###############################################################################
# Tiny finite-horizon SDP example (2 states x 2 actions) for an invasive species
# Case study (toy): coypu / ragondin (Myocastor coypus)
#
# States:
#   "Contained"   = invasion is contained (low abundance / low damage)
#   "Established" = invasion is established (high abundance / high damage)
#
# Actions:
#   1) "Monitor" = do nothing except monitor (cheap, but risk of worsening)
#   2) "Control" = control campaign (costly, but pushes system back to Contained)
#
# Rewards:
#   Reward = -(damage + management_cost)  (so higher = better)
# Horizon:
#   T years, solved by backward induction (dynamic programming).
###############################################################################

#-----------------------------#
# 1) Transition matrices P[[a]]
#-----------------------------#
# P[[a]][s, s'] = Prob( next_state = s' | current_state = s, action = a )
# Rows sum to 1.

states  <- c("Contained", "Established")
actions <- c("Monitor", "Control")

P <- list(
  # a = 1 : Monitor
  # If Contained and only monitoring, there is a chance the invasion spreads / rebounds
  matrix(c(
    0.80, 0.20,   # Contained   -> Contained, Established
    0.00, 1.00    # Established -> Contained, Established (monitoring won't improve)
  ), nrow = 2, byrow = TRUE,
  dimnames = list(states, states)),
  
  # a = 2 : Control
  # A control campaign tends to (partly) reset the system to Contained,
  # even if it was Established (toy "strong action" effect).
  matrix(c(
    0.95, 0.05,   # Contained   -> Contained, Established (rare rebound even after control)
    0.80, 0.20    # Established -> Contained, Established (control often helps, not always)
  ), nrow = 2, byrow = TRUE,
  dimnames = list(states, states))
)

# Quick check (optional): each row should sum to 1
lapply(P, rowSums)

#-----------------------------#
# 2) Reward matrix R[s,a]
#-----------------------------#
# Here we use a toy "net benefit" style reward:
#   reward = -(damage + cost)
#
# Contained: low damage
# Established: high damage
#
# Monitor: low cost
# Control: higher cost

R <- matrix(c(
  # Monitor, Control
  -2,      -6,    # Contained: small damages; control costs more
  -12,     -16    # Established: large damages; control adds cost but may improve next state
), nrow = 2, byrow = TRUE,
dimnames = list(states, actions))

#-----------------------------#
# 3) Backward induction (finite-horizon DP)
#-----------------------------#
T <- 3  # time horizon (e.g., 3 years)

# V[s, t] = optimal value starting from state s at time t
# We'll store V for t = 1..T+1 (T+1 is terminal value = 0 here)
V <- matrix(0, nrow = 2, ncol = T + 1, dimnames = list(states, paste0("t", 1:(T+1))))

# policy[s, t] = optimal action index (1..2) in state s at time t
policy <- matrix(0, nrow = 2, ncol = T, dimnames = list(states, paste0("t", 1:T)))

# Terminal condition: V[, T+1] = 0 (no future reward beyond horizon)
V[, T + 1] <- 0

# Backward loop: compute optimal decisions from t = T down to 1
for (t in T:1) {
  for (s in 1:2) {
    
    # For each action, compute:
    #   Q(a) = immediate reward + expected future value
    # where expected future value = sum_{s'} P[a](s->s') * V[s', t+1]
    Q <- numeric(2)
    
    for (a in 1:2) {
      Q[a] <- R[s, a] + sum(P[[a]][s, ] * V[, t + 1])
    }
    
    # Choose action that maximizes Q
    policy[s, t] <- which.max(Q)
    V[s, t]      <- Q[policy[s, t]]
  }
}

V
policy

# Make the policy more readable (action names instead of indices)
policy_named <- matrix(actions[policy], nrow = 2, dimnames = dimnames(policy))
policy_named

#-----------------------------#
# 4) Same solution using MDPtoolbox (finite horizon)
#-----------------------------#
library(MDPtoolbox)

res <- mdp_finite_horizon(P, R, discount = 1, horizon = T)
res$V
res$policy

# Optional: translate policy indices to action names
actions[res$policy]


# Monitor” est attractif quand le ragondin est contenu, mais il laisse une probabilité de basculer vers ‘Established’.
# “Control” coûte plus cher à court terme, mais réduit le risque futur et peut “ramener” le système vers ‘Contained’.





# Voici une version 3 états (toujours hyper simple) pour le ragondin, avec l’idée :
#   
#   Eradicated : plus de ragondins détectés (mais risque de ré-invasion)
# 
# Contained : présents mais contenus
# 
# Established : bien installés (forts dégâts)
# 
# Deux actions :
#   
#   Monitor (surveiller / pas d’intervention)
# 
# Control (campagne de contrôle)
# 
# L’action Control peut faire passer Established → Contained (souvent) et parfois → Eradicated (rare).
# Même quand c’est “Eradicated”, Monitor peut conduire à une petite probabilité de retour (ré-invasion).


###############################################################################
# Tiny finite-horizon SDP example (3 states x 2 actions) for an invasive species
# Case study (toy): coypu / ragondin (Myocastor coypus)
#
# States:
#   1) "Eradicated"  = no animals detected (could still re-invade)
#   2) "Contained"   = invasion present but contained (low abundance / low damage)
#   3) "Established" = invasion established (high abundance / high damage)
#
# Actions:
#   1) "Monitor" = surveillance only (cheap, but risk of invasion growth / re-invasion)
#   2) "Control" = control campaign (costly, pushes system toward better states)
#
# Rewards:
#   reward = -(damage + management_cost)  (higher is better)
#
# Solved by backward induction (finite-horizon dynamic programming).
###############################################################################

#-----------------------------#
# 1) States & actions
#-----------------------------#
states  <- c("Eradicated", "Contained", "Established")
actions <- c("Monitor", "Control")

S <- length(states)
A <- length(actions)

#-----------------------------#
# 2) Transition matrices P[[a]]
#-----------------------------#
# P[[a]][s, s'] = Prob(next_state = s' | current_state = s, action = a)
# Each row must sum to 1.

P <- list(
  
  # a = 1 : Monitor
  # - If eradicated: small chance of re-invasion to Contained
  # - If contained: some chance it becomes Established without intervention
  # - If established: monitoring alone doesn't improve
  Monitor = matrix(c(
    # to:   Eradicated Contained Established
    0.95,      0.05,     0.00,      # from Eradicated
    0.00,      0.80,     0.20,      # from Contained
    0.00,      0.00,     1.00       # from Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, states)),
  
  # a = 2 : Control
  # - If eradicated: control is mostly unnecessary; still, might reduce re-invasion a bit
  # - If contained: control can sometimes eradicate, often keeps contained
  # - If established: control often improves to contained; rarely eradicates; sometimes fails
  Control = matrix(c(
    # to:   Eradicated Contained Established
    0.98,      0.02,     0.00,      # from Eradicated
    0.20,      0.75,     0.05,      # from Contained
    0.05,      0.75,     0.20       # from Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, states))
)

# Optional check: row sums should be 1
lapply(P, rowSums)

#-----------------------------#
# 3) Reward matrix R[s,a]
#-----------------------------#
# Damage by state (toy):
# - Eradicated: near-zero damage
# - Contained: some damage
# - Established: high damage
#
# Action cost (toy):
# - Monitor: low cost
# - Control: higher cost
#
# reward = -(damage + cost)

damage <- c(Eradicated = 0, Contained = 4, Established = 14)
cost   <- c(Monitor = 1, Control = 6)

R <- outer(damage, cost, FUN = function(d, c) -(d + c))
dimnames(R) <- list(states, actions)
R

#-----------------------------#
# 4) Backward induction (finite horizon)
#-----------------------------#
T <- 5  # horizon (years)

V <- matrix(0, nrow = S, ncol = T + 1,
            dimnames = list(states, paste0("t", 1:(T+1))))
policy <- matrix(0, nrow = S, ncol = T,
                 dimnames = list(states, paste0("t", 1:T)))

# terminal values
V[, T + 1] <- 0

for (t in T:1) {
  for (s in 1:S) {
    
    # compute action values Q(a)
    Q <- numeric(A)
    for (a in 1:A) {
      Q[a] <- R[s, a] + sum(P[[a]][s, ] * V[, t + 1])
    }
    
    # best action
    policy[s, t] <- which.max(Q)
    V[s, t]      <- Q[policy[s, t]]
  }
}

V
policy

# Translate policy indices to action names for readability
policy_named <- matrix(actions[policy], nrow = S, dimnames = dimnames(policy))
policy_named

#-----------------------------#
# 5) Same with MDPtoolbox (finite horizon)
#-----------------------------#
library(MDPtoolbox)

# Note: mdp_finite_horizon expects P as a list (or array) and R as matrix [S x A]
res <- mdp_finite_horizon(P, R, discount = 1, horizon = T)

res$V
res$policy
actions[res$policy]


# 
# Comment lire ce jouet (en 10 secondes)
# 
# Si tu es Established, la valeur future d’un contrôle est forte car tu as une bonne chance de revenir à Contained (voire Eradicated).
# 
# Si tu es Eradicated, Control est rarement utile (coûteux) : on préfère souvent Monitor tant que le risque de ré-invasion reste faible.
# 
# L’horizon compte : plus tu es “près de la fin” (petit T restant), plus tu peux tolérer des actions myopes (éviter un coût de contrôle).
# 
# Si tu veux, je peux te faire une mini-fonction qui simule une trajectoire (états + actions) sous la politique optimale, pour illustrer en cours/diapo.