# https://github.com/boettiger-lab/mdplearning
source("AM_utils.R")

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
A <- length(seq_u)# number of actions



m <- c(0.7, 1.3)
b <- seq(0, 1, 0.02)

# create a dictionary that maps each pair (s_i, b_j) to a unique integer
combinations <- expand.grid(b = b, s = seq_N) # get all combinations
combinations$key <- paste(combinations$b, combinations$s, sep = "_")
dictionary <- setNames(1:nrow(combinations), combinations$key)

step_fun_belief <- function(N, u, m, sigma, b) {
  
  Nnext <- N + r * N * (1 - (N / K) ^ m) - q_u(u) * N
  clamp(Nnext, 0, K)
  
  if (Nnext <= 0) {
    
    P[s, , a] <- c(1, rep(0, S - 1))
    
  } else { # stochastic
    # P(S | s', sigma)
    x <- dlnorm(seq_N, log(Nnext), sdlog = sigma)
    # normalize
    x <- x / sum(x) 
  }
  
  return(x)
}

# get stochastic P
get_P_belief <- function(m, B, S, A, seq_N, seq_u, maxn) {
  
  P <- array(0, dim = c(S * length(b), S * length(b), A))
  
  for (s in seq_len(S)) {
    
    # Current abundance for state index s
    N <- seq_N[s]
    
    for (a in seq_len(A)) {
      
      # Current effort for action index a
      u <- seq_u[a]
      
      for (b in seq_len(length(B))) {
        
        ns <- rep(0, length(B))
        
        for (i in seq_len(maxn)) {
          
          # Continuous next abundance under dynamics + control
          P[s, , a] <- step_fun(N, u, m, sigma) 
          
        }
      }
    }
  }
  
  return(P)
}