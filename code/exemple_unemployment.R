# States X_t: 1 = no job (U), 2 = boring job (B), 3 = good job (G)
# Actions A_t: 1 = decline, 2 = accept good job offer, 3 = accept bad job offer
# Transition matrix Pr(X_t+1|X_t,A_t)
P <- list(
  # a = 1
  # state at t is same state at t + 1
  matrix(c(
    1, 0, 0,
    0, 1, 0,
    0, 0, 1
  ), 3, 3, byrow = TRUE),
  # a = 2
  # accept good offer; move from U at t to G at t + 1 (B/G absorbing states)
  matrix(c(
    0, 0, 1,
    0, 1, 0,
    0, 0, 1
  ), 3, 3, byrow = TRUE),
  # a = 3
  # accept bad offer; move from U at t to B at t + 1 (B/G absorbing)
  matrix(c(
    0, 1, 0,
    0, 1, 0,
    0, 0, 1
  ), 3, 3, byrow = TRUE)
)

# Reward R_t[X_t,A_t]
R <- matrix(
  c(
    # state U ; decline = 0, good = 100, bad = 40
    0, 100,  40,
    # state B : 40 whatever the action (absorbing)
    40,  40,  40,
    # state G : 100 whatever the action (absorbing)
    100, 100, 100
  ),
  nrow = 3,
  byrow = TRUE
)

# Beyond end of horizon, assume no more reward
horizon <- 10
V <- matrix(0, nrow = 3, ncol = horizon + 1) 

# Initialization
policy <- matrix(NA, nrow = 3, ncol = horizon)

# Work backwards in time and compute maximum expected total reward 
# from time 1 to horizon T
for (t in horizon:1) {
  for (s in 1:3) {
    vals <- numeric(3)
    for (a in 1:3) {
      # compute sum of immediate reward we get by taking action a in state s
      # and the expected future value (we weight the known next-step values 
      # by the transition probabilities)
      vals[a] <- R[s, a] + sum(P[[a]][s, ] * V[, t + 1])
    }
    V[s, t] <- max(vals) # whichever action a achieves that maximum
    policy[s, t] <- which.max(vals) # becomes our policy
  }
}

cat("Chaîne des valeurs V[s,t] (états × (horizon+1)) :\n")
print(V)
cat("Politique optimale π[s,t] (états × horizon) :\n")
print(policy)


# ── Version 2 : appel direct à mdp_finite_horizon ────────────────────────────
# (P et R sont déjà définis ci‑dessus)


library(MDPtoolbox)
res <- mdp_finite_horizon(P, R, discount = 1, horizon)
res$V
res$policy


cat("V (dimension (horizon+1) × n_states) :\n")
print(res$V)
cat("π (dimension horizon × n_states) :\n")
print(res$policy)

# Installer DiagrammeR si nécessaire
# install.packages("DiagrammeR")
library(DiagrammeR)

grViz("
digraph backward_induction {
  # Sens : positions de t=9 → t=11 de gauche à droite,
  # flèches indiquant backward induction (de droite à gauche)
  rankdir = LR;

  # Style des nœuds
  node [shape = rectangle, style = rounded, 
        fontname = Helvetica, fontsize = 10, width = 2];

  # Colonne t=9 (gauche)
  subgraph cluster_t9 {
    label = 't = 9';
    V9_U [label = 'V₉(U)\n= 0.5·max(200,70)\n+ 0.5·max(80,70)\n= 140'];
    V9_B [label = 'V₉(B)\n= 40×3 = 120'];
    V9_G [label = 'V₉(G)\n= 100×3 = 300'];
  }

  # Colonne t=10 (milieu)
  subgraph cluster_t10 {
    label = 't = 10';
    V10_U [label = 'V₁₀(U)\n= 0.5·max(100,0)\n+ 0.5·max(40,0)\n= 70'];
    V10_B [label = 'V₁₀(B)\n= 40×2 = 80'];
    V10_G [label = 'V₁₀(G)\n= 100×2 = 200'];
  }

  # Colonne t=11 (droite, terminal)
  subgraph cluster_t11 {
    label = 't = 11';
    V11_U [label = 'V₁₁(U) = 0'];
    V11_B [label = 'V₁₁(B) = 0'];
    V11_G [label = 'V₁₁(G) = 0'];
  }

  # Flèches backward (terminal → t=10 → t=9)
  V11_U -> V10_U -> V9_U [arrowhead = vee];
  V11_B -> V10_B -> V9_B [arrowhead = vee];
  V11_G -> V10_G -> V9_G [arrowhead = vee];
}
")

#   | t  | n = 10−t+1 | Vₜ₊₁(𝐔) | 100·n | 40·n | Décision face à mauvaise offre | Vₜ(𝐔)                                  |
#   | -- | ---------- | -------- | ----- | ---- | ------------------------------ | --------------------------------------- |
#   | 10 | 1          | 0        | 100   | 40   | accepter (40 ≥ 0)              | 0.5·100 + 0.5·40 = **70**               |
#   | 9  | 2          | 70       | 200   | 80   | accepter (80 ≥ 70)             | 0.5·200 + 0.5·80 = **140**              |
#   | 8  | 3          | 140      | 300   | 120  | refuser  (120 < 140)           | 0.5·300 + 0.5·140 = **220**             |
#   | 7  | 4          | 220      | 400   | 160  | refuser  (160 < 220)           | 0.5·400 + 0.5·220 = **310**             |
#   | 6  | 5          | 310      | 500   | 200  | refuser  (200 < 310)           | 0.5·500 + 0.5·310 = **405**             |
#   | 5  | 6          | 405      | 600   | 240  | refuser  (240 < 405)           | 0.5·600 + 0.5·405 = **502.5**           |
#   | 4  | 7          | 502.5    | 700   | 280  | refuser  (280 < 502.5)         | 0.5·700 + 0.5·502.5 = **601.25**        |
#   | 3  | 8          | 601.25   | 800   | 320  | refuser  (320 < 601.25)        | 0.5·800 + 0.5·601.25 = **700.625**      |
#   | 2  | 9          | 700.625  | 900   | 360  | refuser  (360 < 700.625)       | 0.5·900 + 0.5·700.625 = **800.3125**    |
#   | 1  | 10         | 800.3125 | 1000  | 400  | refuser  (400 < 800.3125)      | 0.5·1000 + 0.5·800.3125 = **900.15625** |
#   
# 
# Politique optimale (π*)
# - Si offre bonne → toujours accepter dès la première année.
# - Si offre mauvaise → refuser tant qu’il reste ≥ 3 années (i.e. jusqu’au début de l’année 8).
# À partir de l’année 9, accepter même mauvaise.
# - Si on décline, on se redonne une chance l’année suivante (on recalcule).
# 
# Interprétation
# - Tant que l’horizon restant est long (≥ 3 ans), mieux vaut attendre un bon poste même si on décline de mauvaises offres.
# - Dans les 2 dernières années, on accepte tout, car rester chômeur ne laisse que 1‐2 opportunités pour décrocher un bon poste.
# - En moyenne, la valeur initiale optimale est V₁(𝐔) ≃ 900.16€ sur 10 ans.

