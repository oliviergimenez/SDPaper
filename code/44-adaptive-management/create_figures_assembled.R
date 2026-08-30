library(tidyverse)
library(patchwork)

#############################################
# Common graphical identity
#############################################

col_grey   <- "grey65"
col_dark   <- "grey25"
col_orange <- "#E69F00"
col_red    <- "#D55E00"
col_blue   <- "#0072B2"

theme_sdp <- theme_minimal() +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "bottom"
  )


###################################
###################################
# Structural uncertainty dynamics #
###################################
###################################

# ------------------------------------------------------------
# Population dynamics
# ------------------------------------------------------------

r <- 0.9
K <- 2000

seq_N <- seq(
  0,
  K,
  by = 20
)

# Candidate model set
m <- c(
  0.4,
  0.7,
  1,
  1.15,
  1.3,
  1.7
)

# True model
true_m <- m[4]


# ------------------------------------------------------------
# Build model trajectories
# ------------------------------------------------------------

m_data <- data.frame(
  N = NULL,
  R = NULL,
  m = NULL
)

for (i in seq_along(m)) {
  
  m_data <- rbind(
    m_data,
    data.frame(
      N = seq_N,
      R = r * (1 - (seq_N / K)^m[i]),
      m = rep(
        m[i],
        length(seq_N)
      )
    )
  )
}


# ------------------------------------------------------------
# Consistent colours for candidate models
# ------------------------------------------------------------

model_cols <- c(
  "0.4"  = "grey70",
  "0.7"  = "#56B4E9",
  "1"    = "#0072B2",
  "1.15" = col_red,
  "1.3"  = col_orange,
  "1.7"  = "grey30"
)

model_labels <- c(
  "0.4"  = "0.4",
  "0.7"  = "0.7",
  "1"    = "1",
  "1.15" = "1.15 (true model)",
  "1.3"  = "1.3",
  "1.7"  = "1.7"
)


# ------------------------------------------------------------
# A. Candidate model set
# ------------------------------------------------------------

m_plot <- ggplot() +
  
  # Other candidate models
  geom_line(
    data = m_data[m_data$m != true_m, ],
    aes(
      x = N,
      y = R,
      colour = as.factor(m)
    ),
    linewidth = 0.9
  ) +
  
  # True model
  geom_line(
    data = m_data[m_data$m == true_m, ],
    aes(
      x = N,
      y = R,
      colour = as.factor(m)
    ),
    linetype = "dashed",
    linewidth = 1.2
  ) +
  
  scale_colour_manual(
    values = model_cols,
    breaks = names(model_cols),
    labels = model_labels
  ) +
  
  labs(
    x = "Abundance N",
    y = "R(N)",
    colour = "m"
  ) +
  
  # Legend will be shown only once, in panel B
  guides(
    colour = "none"
  ) +
  
  theme_sdp +
  
  ggtitle("A.")

m_plot


# ------------------------------------------------------------
# Read policies assuming known dynamics
# ------------------------------------------------------------

known_policies <- read.csv(
  "../SDPaper/code/44-adaptive-management/data/passive_known_dynamics.csv"
)


# ------------------------------------------------------------
# B. Optimal policies with known dynamics
# ------------------------------------------------------------

plot_known_policies <- ggplot() +
  
  geom_line(
    data = known_policies[known_policies$m != true_m, ],
    aes(
      x = N,
      y = policy,
      colour = as.factor(m)
    ),
    linewidth = 0.9
  ) +
  
  geom_line(
    data = known_policies[known_policies$m == true_m, ],
    aes(
      x = N,
      y = policy,
      colour = as.factor(m)
    ),
    linetype = "dashed",
    linewidth = 1.2
  ) +
  
  scale_colour_manual(
    values = model_cols,
    breaks = names(model_cols),
    labels = model_labels
  ) +
  
  labs(
    x = "Abundance N",
    y = "Optimal effort",
    colour = "m"
  ) +
  
  theme_sdp +
  
  ggtitle("B.")

plot_known_policies



###############################
###############################
# Passive adaptive management #
###############################
###############################

# ------------------------------------------------------------
# Read passive adaptive-management results
# ------------------------------------------------------------

posterior <- read.csv(
  "../SDPaper/code/44-adaptive-management/data/passive_am_belief.csv"
)[, -1]

# Posterior beliefs are available for the five alternative models
colnames(posterior) <- m[c(1:3, 5, 6)]

am_data <- read.csv(
  "../SDPaper/code/44-adaptive-management/data/passive_am_data.csv"
)


# ------------------------------------------------------------
# Convert posterior model probabilities to long format
# ------------------------------------------------------------

Tmax <- 30

posterior_long <- posterior %>%
  mutate(
    t = 1:Tmax
  ) %>%
  pivot_longer(
    cols = -t,
    names_to = "m",
    values_to = "belief"
  )


# ------------------------------------------------------------
# C. Passive adaptive management: model beliefs
# ------------------------------------------------------------

plot_belief <- ggplot(
  posterior_long,
  aes(
    x = t,
    y = belief,
    colour = as.factor(m)
  )
) +
  
  geom_line(
    linewidth = 1
  ) +
  
  scale_colour_manual(
    values = model_cols
  ) +
  
  labs(
    x = "Time step",
    y = "Model belief",
    colour = "m"
  ) +
  
  # Suppress here to avoid a second legend
  guides(
    colour = "none"
  ) +
  
  theme_sdp +
  
  ggtitle("C.")

plot_belief


# ------------------------------------------------------------
# D. Passive adaptive management: reward
# ------------------------------------------------------------

plot_reward <- ggplot(
  am_data,
  aes(
    x = time,
    y = value
  )
) +
  
  geom_line(
    colour = col_blue,
    linewidth = 1.1
  ) +
  
  labs(
    x = "Time step",
    y = "Reward"
  ) +
  
  theme_sdp +
  
  theme(
    legend.position = "none"
  ) +
  
  ggtitle("D.")

plot_reward



##############################
##############################
# Active adaptive management #
##############################
##############################

# ------------------------------------------------------------
# State space used for active AM
# ------------------------------------------------------------

seq_N_am <- seq(
  0,
  K,
  by = 100
)


# ------------------------------------------------------------
# Read active adaptive-management results
# ------------------------------------------------------------

discount_high <- read.csv(
  "../SDPaper/code/44-adaptive-management/data/active_am_data_highdiscount.csv"
) %>%
  mutate(
    N = seq_N_am
  )

discount_low <- read.csv(
  "../SDPaper/code/44-adaptive-management/data/active_am_data_lowdiscount.csv"
) %>%
  mutate(
    N = seq_N_am
  )


# ------------------------------------------------------------
# E. Active AM: gamma = 0.95
# ------------------------------------------------------------

plot_high <- ggplot(
  discount_high,
  aes(
    x = N,
    y = policy
  )
) +
  
  geom_line(
    colour = col_blue,
    linewidth = 1.1
  ) +
  
  annotate(
    "text",
    x = Inf,
    y = Inf,
    label = expression(gamma == 0.95),
    hjust = 1.5,
    vjust = 2.1,
    size = 5
  ) +
  
  labs(
    x = "Abundance N",
    y = "Optimal action"
  ) +
  
  theme_sdp +
  
  theme(
    legend.position = "none"
  ) +
  
  ggtitle("E.")

plot_high


# ------------------------------------------------------------
# F. Active AM: gamma = 0.75
# ------------------------------------------------------------

plot_low <- ggplot(
  discount_low,
  aes(
    x = N,
    y = policy
  )
) +
  
  geom_line(
    colour = col_blue,
    linewidth = 1.1
  ) +
  
  annotate(
    "text",
    x = Inf,
    y = Inf,
    label = expression(gamma == 0.75),
    hjust = 1.5,
    vjust = 2.1,
    size = 5
  ) +
  
  labs(
    x = "Abundance N",
    y = "Optimal action"
  ) +
  
  theme_sdp +
  
  theme(
    legend.position = "none"
  ) +
  
  ggtitle("F.")

plot_low



#############################################
#############################################
# Assemble complete adaptive-management fig #
#############################################
#############################################

figure_adaptive <- (
  
  m_plot + plot_known_policies
  
) / (
  
  plot_belief + plot_reward
  
) / (
  
  plot_high + plot_low
  
) +
  
  plot_layout(
    guides = "collect"
  ) &
  
  theme(
    legend.position = "bottom"
  )


# ------------------------------------------------------------
# Display
# ------------------------------------------------------------

figure_adaptive


# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

ggsave(
  "../SDPaper/figures/figure5.png",
  figure_adaptive,
  dpi = 600,
  width = 9,
  height = 10
)


