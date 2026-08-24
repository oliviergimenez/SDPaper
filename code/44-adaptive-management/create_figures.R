library(tidyverse)
library(patchwork)
library(viridis)

###################################
###################################
# Structural uncertainty dynamics #
###################################
###################################

# get logistic growth parameter values
r <- 0.9 # intrinsic rate of growth
K <- 2000 # carrying capacity
seq_N <- seq(0, K, by = 20) # state space

# model set
m <- c(0.4, 0.7, 1, 1.15, 1.3, 1.7)
true_m <- m[4]

# set up empty df
m_data <- data.frame(
  N = NULL, R = NULL, m = NULL
)

for (i in seq_along(m)) {
  m_data <- rbind(m_data,
                  data.frame(
                    N = seq_N, 
                    R = r * (1 - (seq_N / K) ^ m[i]),
                    m = rep(m[i], length(seq_N))
                  ))
}

m_plot <- ggplot() +
  geom_line(data = m_data[m_data$m != true_m, ],
            aes(x = N, y = R, color = as.factor(m))) +
  labs(x = "N", y = "R(N)", color = "m") +
  geom_line(data = m_data[m_data$m == true_m, ],
            aes(x = N, y = R, color = "True Model"),
            linetype = "dashed") +
  scale_color_manual(values = c(
    "True Model" = "firebrick",
    setNames(viridis_pal()(length(unique(m_data$m)) - 1), 
             as.character(unique(m_data$m[m_data$m != true_m])))
  )) +
  ggtitle("A. Model set") +
  theme_minimal() +
  theme(legend.position = "None")

##############################
# policies with known dynamics

# read in data
known_policies <- read.csv("data/passive_known_dynamics.csv")

# plot
plot_known_policies <- ggplot() +
  geom_line(data = known_policies[known_policies$m != true_m, ],
            aes(x = N, y = policy, color = as.factor(m))) +
  labs(x = "N", y = "optimal effort", color = "m") +
  geom_line(data = known_policies[known_policies$m == true_m, ],
            aes(x = N, y = policy, color = "True Model"),
            linetype = "dashed") +
  scale_color_manual(values = c(
    "True Model" = "firebrick",
    setNames(viridis_pal()(length(unique(known_policies$m)) - 1), 
             as.character(unique(known_policies$m[known_policies$m != true_m])))
  )) +
  ggtitle("B. Optimal policies\nwith known dynamics") +
  theme_minimal()

final_plot <- m_plot + plot_known_policies + plot_layout(nrow = 1)

# save
ggsave("figures/model_set_policies.png", final_plot, dpi = 400,
       width = 7, height = 3)


###############################
###############################
# Passive adaptive management #
###############################
###############################

# read in data
posterior <- read.csv("data/passive_am_belief.csv")[, -1]
colnames(posterior) <- m[c(1:3, 5, 6)]
am_data <- read.csv("data/passive_am_data.csv")

# convert posterior to long
Tmax <- 30
posterior_long <- posterior %>% 
  mutate(t = 1:Tmax) %>% 
  pivot_longer(cols = -t,
               names_to = "m",
               values_to = "belief")

# plot
plot_belief <- ggplot(data = posterior_long) +
  geom_line(aes(x = t, y = belief, color = as.factor(m))) +
  scale_color_viridis_d() +
  ggtitle("A.") +
  labs(x = "t", y = "belief", color = "m") +
  theme_minimal()

plot_reward <- ggplot(data = am_data) +
  geom_line(aes(x = time, y = value)) +
  ggtitle("B.") +
  labs(x = "t", y = "reward") +
  theme_minimal()

plot_passive <- plot_belief + plot_reward + plot_layout(ncol = 1)

# save
ggsave("figures/passive_AM.png", plot_passive, dpi = 400,
       width = 4, height = 5)


##############################
##############################
# Active adaptive management #
##############################
##############################

# get states
seq_N_am <- seq(0, K, by = 100)

# read in data
discount_high <- read.csv("data/active_am_data_highdiscount.csv") %>% 
  mutate(N = seq_N_am)
discount_low <- read.csv("data/active_am_data_lowdiscount.csv") %>% 
  mutate(N = seq_N_am)

# plot
plot_high <- ggplot(data = discount_high) +
  geom_line(aes(x = N, y = policy)) +
  labs(x = "N", y = "optimal action") +
  ggtitle("A. Discount factor: 0.95") +
  theme_minimal()

plot_low <- ggplot(data = discount_low) +
  geom_line(aes(x = N, y = policy)) +
  labs(x = "N", y = "optimal action") +
  ggtitle("B. Discount factor: 0.75") +
  theme_minimal()  

final_active <- plot_high + plot_low + plot_layout(nrow = 1)
ggsave("figures/active_AM.png", final_active, dpi = 400,
       width = 6, height = 3)
