# app.R

library(shiny)
library(igraph)
library(tidyverse)   # for pivot_longer(), etc.
library(ggrepel)

# ──────────────────────────────────────────────────────────────────────────
# 1) SOURCE  MDP + forward‐simulation to get:
#     g         : igraph network of n_sites nodes
#     state_mat : Tmax × n_sites matrix of 0/1 occupancy under optimal policy
#     pop_mat   : Tmax × n_sites matrix of abundances under optimal policy
#     villes    : character vector of site names, length = n_sites
#     Tmax      : integer number of time steps
# ──────────────────────────────────────────────────────────────────────────
#source("SDP-coypu.R")

# ──────────────────────────────────────────────────────────────────────────
# 2) FIXED LAYOUT + CONVERT TO tbl_graph
# ──────────────────────────────────────────────────────────────────────────
coords <- layout_with_fr(g)
V(g)$x    <- coords[,1]
V(g)$y    <- coords[,2]
V(g)$name <- villes

nodes <- tibble(
  name = villes,
  x    = coords[,1],
  y    = coords[,2]
)
edges_df <- igraph::as_data_frame(g, what = "edges") %>%
  transmute(
    from_x = nodes$x[from],
    from_y = nodes$y[from],
    to_x   = nodes$x[to],
    to_y   = nodes$y[to]
  )

# also compute max_pop once
max_pop <- max(pop_mat, na.rm = TRUE)

# ──────────────────────────────────────────────────────────────────────────
# 3) UI
# ──────────────────────────────────────────────────────────────────────────
ui <- fluidPage(
  titlePanel("Coypu Network Dynamics under MDP‐Optimal Policy"),
  sidebarLayout(
    sidebarPanel(
      sliderInput(
        "time", "Time step",
        min     = 1,
        max     = Tmax,
        value   = 1,
        step    = 1,
        animate = animationOptions(interval = 800, loop = TRUE)
      )
    ),
    mainPanel(
      plotOutput("netPlot", height = "600px")
    )
  )
)

# ──────────────────────────────────────────────────────────────────────────
# 4) SERVER
# ──────────────────────────────────────────────────────────────────────────
server <- function(input, output, session) {
  output$netPlot <- renderPlot({
    t <- input$time
    
    # 1) current occupancy & abundance
    st  <- state_mat[t, ]
    pop <- pop_mat[t, ]
    
    # 2) build nodes_now
    nodes_now <- nodes %>%
      mutate(
        status   = factor(st, levels = c(0,1),
                          labels = c("empty","occupied")),
        pop_size = pop
      )
    
    # 3) draw with ggplot2
    ggplot() +
      # fixed edges
      geom_segment(data = edges_df,
                   aes(x = from_x, y = from_y,
                       xend = to_x,   yend = to_y),
                   color = "grey80") +
      # nodes
      geom_point(data = nodes_now,
                 aes(x = x, y = y,
                     size = pop_size,
                     fill = status),
                 shape = 21, color = "black", stroke = 0.3) +
      # labels
      geom_text_repel(
        data = nodes_now,
        aes(x = x, y = y, label = name),
        size = 3,
        min.segment.length = 0,    # draw segments even if short
        box.padding        = 0.5,
        point.padding      = 0.5
      ) +      # status fill
      scale_fill_manual(
        name   = "Status",
        values = c(empty = "white", occupied = "forestgreen")
      ) +
      # locked size legend
      # binned size legend with 5 breaks and real counts
        scale_size_continuous(
          name   = "Abundance",
          range  = c(5, 15),
          limits = c(0, max_pop),
          breaks = seq(0, max_pop, length.out = 5),
          labels = round(seq(0, max_pop, length.out = 5))
        ) +
      coord_equal(clip = "off") +
      scale_x_continuous(expand = expansion(mult = c(0.1, 0.1))) +
      scale_y_continuous(expand = expansion(mult = c(0.1, 0.1))) +
      labs(subtitle = paste("Time step:", t)) +
      theme_void() +
      theme(
        plot.subtitle        = element_text(face = "italic", hjust = 0.5),
        legend.position      = c(1.15, 0.5),       # x > 1 pushes outside
        legend.justification = c(0, 0.5),          # anchor legend’s left middle
        plot.margin          = margin(5, 80, 5, 5),# add right space for legend
        legend.background    = element_rect(fill = alpha("white", 0.6)),
        legend.key.size      = unit(0.8, "lines")      )
  })
}

# ──────────────────────────────────────────────────────────────────────────
# 5) RUN APP
# ──────────────────────────────────────────────────────────────────────────
shinyApp(ui, server)
