###############################################################################
# Tiny finite-horizon POMDP example (3 states x 3 actions) for an invasive species
# Case study (toy): coypu / ragondin (Myocastor coypus)
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
#   1) "Absent"  = coypu not observed
#   2) "Present" = coypu observed
#
# Rewards:
#   Reward = -(damage + management_cost)  (so higher = better)
#
# Horizon:
#   Explore various.
#
# Note: Below P, O, and R have been formatted to suit pomdp and sarsop packages
# sarsop only compiles in VS Code
###############################################################################
install.packages("pomdp")
library(pomdp)
install.packages("sarsop")
library(sarsop)

states  <- c("Eradicated", "Contained", "Established"); S <- length(states)
actions <- c("DoNothing", "FertControl", "LethControl"); A <- length(actions)
observations = c("Absent", "Present")

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
P_sarsop <- array(c(DoNothing, FertControl, LethControl), dim = c(S, S, A))

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

R_pomdp <- rbind(
  R_("DoNothing",   "Eradicated",  v = reward[1,1]),
  R_("DoNothing",   "Contained",   v = reward[2,1]),
  R_("DoNothing",   "Established", v = reward[3,1]),
  R_("FertControl", "Eradicated",  v = reward[1,2]),
  R_("FertControl", "Contained",   v = reward[2,2]),
  R_("FertControl", "Established", v = reward[3,2]),
  R_("LethControl", "Eradicated",  v = reward[1,3]),
  R_("LethControl", "Contained",   v = reward[2,3]),
  R_("LethControl", "Established", v = reward[3,3])
)

R_sarsop <- reward

#-----------------------------#
# 4) Detection probabilities O[s,a]
#-----------------------------#
# Here we assume that detectability is detendent on the state of the system and the action take.
# That is, we assume that 'DoNothing' reduces our ability to detect coypu, while both control measures
# have the same detection probabilities.
# Detection increases with the relative abundance of the coypu population. 

DoNothing = matrix(c(
  # Absent   Present
  0.70, 0.30, # Detectability when Eradicated
  0.60, 0.40,  # Detectability when Contained
  0.40, 0.60 # Detectability when Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, observations)
)

FertControl = matrix(c(
  # Absent   Present
  0.05, 0.95, # Detectability when Eradicated
  0.40, 0.60,  # Detectability when Contained
  0.15, 0.85 # Detectability when Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, observations)
)

LethControl = matrix(c(
  # Absent   Present
  0.05, 0.95, # Detectability when Eradicated
  0.40, 0.60,  # Detectability when Contained
  0.15, 0.85 # Detectability when Established
  ), nrow = S, byrow = TRUE, dimnames = list(states, observations)
)

O_pomdp <- list(DoNothing, FertControl, LethControl)
O_sarsop <- array(c(DoNothing, FertControl, LethControl), dim = c(S, S, A))


#-----------------------------#
# 5) Solve POMDPs
#-----------------------------#
b0 <- c(0,0,1)
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

sol_sarsop <- sarsop(
  transition = P_sarsop, 
  observation = O_sarsop, 
  reward = R_sarsop, 
  discount = discount,
  state_prior	= b0
)

# Inspect outputs:
# The first columns of the data.table returned by policy() provide the α-vectors coefficient (one per line), the following column provides the optimal action
policy(sol_pomdp)

# each column corresponds to an alpha vector
# each line corresponds to a state
sol_sarsop$vectors # alpha vectors

# Optimal value function and policy for a belief state b:
b <- c(0.10,0.15,0.75) # belief state

# using pomdp output for finite time
alphas_pomdp <- policy(sol_pomdp)[[1]][,c(1,2,3)] # alpha vectors
a_pomdp <- b %*% t(alphas_pomdp) # dot product between b and each alpha vector
optimal_pomdp <- list(max(a_pomdp), policy(sol_pomdp)[[1]][which.max(a_pomdp), 4]) # alpha vector and its action that maximise the value function
optimal_pomdp

# using pomdp output for infinite time
alphas_pomdp <- policy(sol_pomdp)[,c(1,2,3)] # alpha vectors
a_pomdp <- b %*% t(alphas_pomdp) # dot product between b and each alpha vector
optimal_pomdp <- list(max(a_pomdp), policy(sol_pomdp)[which.max(a_pomdp), 4]) # alpha vector and its action that maximise the value function
optimal_pomdp

# using sarsop output
a_sarsop <- b %*% sol_sarsop$vectors # dot product between b and each alpha vector
optimal_sarsop <- list(max(a_sarsop), sol_sarsop$action[which.max(a_sarsop)]) # alpha vector and its action that maximise the value function
optimal_sarsop

# Plot the policy graph:
plot_policy_graph(sol_pomdp, engine="visNetwork")