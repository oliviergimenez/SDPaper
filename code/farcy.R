# Installer MDPtoolbox si nécessaire
# install.packages("MDPtoolbox")
library(MDPtoolbox)

# Préparer l’affichage : 1 ligne × 3 colonnes
par(mfrow = c(1, 3))

#### Figure 2a
a <- 1  # Gompertz
b <- 5  # Gompertz
c <- 2  # Gompertz
x <- seq(0, 5, by = 0.01)
q1 <- a * exp(-b * exp(-c * x))
plot(x, q1, type = "l",
     xlab = "Expenditure, x",
     ylab = "Perfection of knowledge, p")

#### Figure 2b
# Matrice des gains
M <- matrix(c(4, -2,
              -5,  5),
            nrow = 2, byrow = TRUE)
q <- 0.5  # probabilité que la nature soit "bonne"

# Calcul de l’action optimale s (1 ou 2)
payoff1 <- q * M[1,1] + (1 - q) * M[1,2]
payoff2 <- q * M[2,1] + (1 - q) * M[2,2]
s <- if (payoff1 >= payoff2) 1 else 2

# Valeur parfaite et EVPI
p <- 1
perfect <- q * p * max(M[,1]) +
  q * (1 - p) * M[s,1] +
  (1 - q) * p * max(M[,2]) +
  (1 - q) * (1 - p) * M[s,2]
EVPI <- perfect - (q * M[s,1] + (1 - q) * M[s,2])

# Fonction de valeur nette v(x)
v <- function(x, M, s, q, a, b, c) {
  info_perf <- a * exp(-b * exp(-c * x))
  q * info_perf * max(M[,1]) +
    q * (1 - info_perf) * M[s,1] +
    (1 - q) * info_perf * max(M[,2]) +
    (1 - q) * (1 - info_perf) * M[s,2] -
    x
}

# Tracé de v(x)
plot(x, v(x, M, s, q, a, b, c), type = "l",
     xlab = "Expenditure, x",
     ylab = "Value")

#### Figure 2c
# Même M, nouveaux paramètres Gompertz
M <- matrix(c(4, -2,
              -5,  5),
            nrow = 2, byrow = TRUE)
a <- 1; b <- 6; c <- 1
qi <- seq(0, 1, by = 0.01)
s_vec <- integer(length(qi))
vi    <- numeric(length(qi))

for (i in seq_along(qi)) {
  q_i <- qi[i]
  # choix de s pour ce q_i
  payoffs <- c(q_i * M[1,1] + (1 - q_i) * M[1,2],
               q_i * M[2,1] + (1 - q_i) * M[2,2])
  s_vec[i] <- which.max(payoffs)
  # optimiser x ↦ v(x) sur [0,5]
  opt <- optimize(function(x) -v(x, M, s_vec[i], q_i, a, b, c),
                  interval = c(0, 5))
  vi[i] <- opt$minimum
}

plot(qi, vi, type = "l",
     xlab = 'Pr[Nature is "good"]',
     ylab = "Optimal expenditure")

#### Figure 3 (horizon fini, 4 états × 4 actions)
g1 <- 0.2; g2 <- 0.2
th1 <- 0.3; th2 <- 0.2

# Matrices de transition T0 à T3 (4×4)
T0 <- matrix(c((1-g1)*(1-g2), g1*(1-g2),   (1-g1)*g2,   g1*g2,
               0,               (1-g2),       0,           g2,
               0,               0,            (1-g1),      g1,
               0,               0,            0,           1),
             nrow = 4, byrow = TRUE)

T1 <- matrix(c(
  g1*th1*(1-g2)+(1-g1)*(1-g2), g1*(1-th1)*(1-g2),     g2*(1-g1)+g2*g1*th1,       g1*(1-th1)*g2,
  th1*(1-g2),                  (1-th1)*(1-g2),        th1*g2,                    (1-th1)*g2,
  0,                           0,                      (1-g1)+g1*th1,             g1*(1-th1),
  0,                           0,                      th1,                       (1-th1)
), nrow = 4, byrow = TRUE)

T2 <- matrix(c(
  g2*th2*(1-g1)+(1-g1)*(1-g2), g1*g2*th2+g1*(1-g2),   g2*(1-th2)*(1-g1),         g1*g2*(1-th2),
  0,                           g2*th2+(1-g2),        0,                          g2*(1-th2),
  (1-g1)*th2,                  g1*th2,                (1-g1)*(1-th2),            g1*(1-th2),
  0,                           th2,                   0,                          1-th2
), nrow = 4, byrow = TRUE)

T3 <- matrix(c(
  g1*th1*g2*th2 + (1-g1)*(1-g2) + g1*th1*(1-g2) + g2*th2*(1-g1),
  g1*(1-th1)*(1-g2) + g1*(1-th1)*g2*th2,
  g2*(1-th2)*(1-g1) + g2*(1-th2)*g1*th1,
  g1*(1-th1)*g2*(1-th2),
  
  g2*th2*th1 + th1*(1-g2),
  (1-g2)*(1-th1) + g2*th2*(1-th1),
  th1*g2*(1-th2),
  g2*(1-th2)*(1-th1),
  
  (1-g1)*th2 + g1*th1*th2,
  th2*g1*(1-th1),
  g1*th1*(1-th2) + (1-g1)*(1-th2),
  g1*(1-th1)*(1-th2),
  
  th1*th2,
  th2*(1-th1),
  th1*(1-th2),
  (1-th1)*(1-th2)
), nrow = 4, byrow = TRUE)

# Vérification : chaque ligne doit sommer à 1
rowSums(T0); rowSums(T1); rowSums(T2); rowSums(T3)

# Récompenses R (4 états × 4 actions)
s1 <- -3  # traiter état 1
s2 <- -4  # traiter état 2
s3 <- -10 # laisser état 1
s4 <- -12 # laisser état 2
s5 <- (s1 + s3) * 1.25  # synergie

R_mat <- cbind(
  c(0,    s3,      s4,      s5),
  c(s1,   s1 + s3, s1 + s4, s1 + s5),
  c(s2,   s2 + s3, s2 + s4, s2 + s5),
  c(s1+s2, s1+s2+s3, s1+s2+s4, s1+s2+s5)
)

# Algorithme à horizon fini
prob_list <- list(T0, T1, T2, T3)
discount  <- 0.95
horizon   <- 20

res <- mdp_finite_horizon(prob_list, R_mat, discount, horizon)
policy <- res$policy  # matrice (horizon × n_states)

# Affichage de la politique (heatmap)
image(x = 1:ncol(policy),
      y = 1:nrow(policy),
      z = t(policy[nrow(policy):1, ]),
      xlab = "État",
      ylab = "Période",
      main = "Politique optimale")
axis(1, at = 1:ncol(policy))
axis(2, at = 1:nrow(policy), labels = rev(1:nrow(policy)))
