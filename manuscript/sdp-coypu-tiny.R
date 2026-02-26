# -------------------------------------------------------------------------
# Tiny stochastic dynamic programming example for an invasive species
# -------------------------------------------------------------------------
# Goal:
# Illustrate how a finite-horizon Markov Decision Process (MDP) can be used
# to determine optimal management decisions (Monitor vs Control)
# depending on the invasion state (Eradicated, Contained, Established).
# -------------------------------------------------------------------------

# -----------------------------
# 1) Define states and actions
# -----------------------------

# Ecological states describing invasion level
states <- c("Eradicated", "Contained", "Established")

# Management actions available at each time step
actions <- c("Monitor", "Control")


# -----------------------------
# 2) Transition matrices P_a
# -----------------------------
# P[[a]][s, s'] = Prob(next state = s' | current state = s, action = a)
# Rows sum to 1 because they define probability distributions.

P <- list(
  
  # Dynamics under monitoring (no intervention)
  Monitor = matrix(c(
    0.95, 0.05, 0.00,   # From Eradicated → mostly stays eradicated,
    # small chance of re-invasion
    0.00, 0.80, 0.20,   # From Contained → may worsen to Established
    0.00, 0.00, 1.00    # From Established → cannot improve without control
  ), nrow = 3, byrow = TRUE,
  dimnames = list(states, states)),
  
  # Dynamics under control intervention
  Control = matrix(c(
    0.98, 0.02, 0.00,   # From Eradicated → control slightly reduces reinvasion risk
    0.20, 0.75, 0.05,   # From Contained → may eradicate or stay contained
    0.05, 0.75, 0.20    # From Established → usually improves but not always
  ), nrow = 3, byrow = TRUE,
  dimnames = list(states, states))
)

# Optional check: each row should sum to 1 (valid probability distributions)
lapply(P, rowSums)


# -----------------------------
# 3) Reward matrix R(s,a)
# -----------------------------
# Rewards are defined as negative total costs:
# reward = -(damage + management cost)

# Ecological damage associated with each state
damage <- c(Eradicated = 0, Contained = 4, Established = 14)

# Cost of each management action
cost   <- c(Monitor = 1, Control = 6)

# Construct matrix of rewards for all state–action combinations
R <- outer(damage, cost, function(d, c) -(d + c))
dimnames(R) <- list(states, actions)

R


# -------------------------------------------------------------------------
# 4) Finite-horizon dynamic programming (manual backward induction)
# -------------------------------------------------------------------------
# We solve the decision problem over a finite horizon T.

T <- 5                     # number of decision steps
S <- length(states)        # number of states
A <- length(actions)       # number of actions

# V[s, t] = optimal expected cumulative reward
# when starting in state s at time t
# The last column (t = T+1) is terminal value = 0
V <- matrix(0, nrow = S, ncol = T + 1,
            dimnames = list(states, paste0("t", 1:(T+1))))

# policy[s, t] = index of optimal action at state s and time t
policy <- matrix(0, nrow = S, ncol = T,
                 dimnames = list(states, paste0("t", 1:T)))


# Backward induction:
# We compute optimal decisions starting from the end of the horizon
for (t in T:1) {
  for (s in 1:S) {
    
    # Q(a) = value of taking action a in state s at time t
    # = immediate reward + expected future value
    Q <- numeric(A)
    
    for (a in 1:A) {
      Q[a] <- R[s, a] + sum(P[[a]][s, ] * V[, t + 1])
    }
    
    # Choose action with highest expected value
    policy[s, t] <- which.max(Q)
    V[s, t]      <- Q[policy[s, t]]
  }
}

# Optimal value function
V

# Convert action indices into action names
policy_named <- matrix(actions[policy], nrow = S,
                       dimnames = dimnames(policy))
policy_named


# -------------------------------------------------------------------------
# 5) Solve the same problem with MDPtoolbox
# -------------------------------------------------------------------------
# This provides a built-in implementation of finite-horizon SDP.

library(MDPtoolbox)

res <- mdp_finite_horizon(P, R, discount = 1, N = T)

# Value function returned by the package
res$V

# Optimal policy (action index → convert to labels)
actions[res$policy]
