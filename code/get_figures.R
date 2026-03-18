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
  ))

# save
ggsave("figures/model_set.png", m_plot, dpi = 400,
       width = 5, height = 3)

###############################
###############################
# Passive adaptive management #
###############################
###############################

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
  ))

# save
ggsave("figures/passive_known_policies.png", plot_known_policies, dpi = 400,
       width = 5, height = 3)


########################
# passive AM simulations 

# read in data
posterior <- read.csv("data/passive_am_belief.csv")[, -1]
colnames(posterior) <- m[c(1:3, 5, 6)]
am_data <- read.csv("data/passive_am_data.csv")

# convert posterior to long
posterior_long <- posterior %>% 
  mutate(t = 1:Tmax) %>% 
  pivot_longer(cols = -t,
               names_to = "m",
               values_to = "belief")

# plot
plot_belief <- ggplot(data = posterior_long) +
  geom_line(aes(x = t, y = belief, color = as.factor(m))) +
  scale_color_viridis_d() +
  labs(x = "t", y = "belief", color = "m")

plot_reward <- ggplot(data = df) +
  geom_line(aes(x = time, y = value)) +
  labs(x = "t", y = "reward")

plot_passive <- plot_belief + plot_reward + plot_layout(ncol = 1)

# save
ggsave("figures/passive_AM.png", plot_passive, dpi = 400,
       width = 4, height = 5)
