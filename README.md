## Using stochastic dynamic programming for making decisions about invasive species

This repository contains the code, figures, and manuscript accompanying the paper by Gimenez, Keller, Marescot, Boettiger and Speakman.

> **Using stochastic dynamic programming for making decisions about invasive species**

The paper provides a practical introduction to stochastic dynamic programming (SDP) for invasive species management, with progressively more complex worked examples covering:

- single-population management,
- spatially structured populations,
- partially observable Markov decision processes (POMDPs),
- value of information and adaptive management (passive and active).

All analyses are fully reproducible in `R` using the scripts provided in this repository.

---

## Repository structure

```
.
├── code/
│   ├── 2-explanation-of-the-method: toy example used in section 2
│   ├── 41-single-population: single population example used in section 4.1
│   ├── 42-connected-populations: example with connected populations used in section 4.2
│   └── 43-imperfect-detection: pomdp example used in section 4.3
│   └── 44-adaptive-management: adaptive management examples used in section 4.4
│
├── figures/
│   └── Figures used in the manuscript
│
└── manuscript/
    ├── tuto-sdp-invasive-species.Rmd: R Markdown manuscript
    ├── references.bib: bibliography
    └── ...
```

