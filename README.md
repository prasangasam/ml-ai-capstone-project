# BBO Capstone Project – Adaptive Black-Box Optimisation

## Repository

https://github.com/prasangasam/ml-ai-capstone-project

## Project Overview

This project focuses on solving black-box optimisation problems using adaptive optimisation and reinforcement learning-inspired strategies. The goal was to identify high-performing query points for eight unknown functions across multiple weekly optimisation rounds.

The optimisation strategy evolved over time:

- Early stages focused on exploration
- Later stages shifted towards exploitation and adaptive refinement
- Weekly feedback was used to improve query generation and convergence

## Non-Technical Summary

This project explored how AI can solve optimisation problems when the exact mathematical functions are unknown. Using reinforcement learning-inspired strategies, the system gradually learned where better solutions were likely to exist by analysing feedback from previous attempts. Early stages focused on exploration to gather information, while later stages concentrated on refining the best-performing areas. Over multiple weeks, the optimisation process became more efficient and adaptive. The project demonstrates how AI techniques such as exploration, exploitation and feedback-driven learning can support optimisation in logistics, resilience and engineering systems.

## Final Leaderboard Positions

| Function   | Position |
| ---------- | -------- |
| Function 1 | 37       |
| Function 2 | 8        |
| Function 3 | 31       |
| Function 4 | 9        |
| Function 5 | 19       |
| Function 6 | 23       |
| Function 7 | 12       |
| Function 8 | 15       |

## Weekly Optimisation Improvements

- Function 5 improved from approximately 49.85 to 8662.40
- Function 8 improved from 7.94 to 9.96
- Function 7 improved from 0.46 to above 2.8

## Technologies Used

- Python
- NumPy
- Pandas
- Jupyter Notebook
- Reinforcement Learning Concepts

## Key Lessons

- Exploration is essential during early optimisation
- Exploitation improves convergence later
- Feedback-driven optimisation improves performance over time

## Repository Structure

```
ml-ai-capstone-project/
│
├── scripts/
│   ├── run_week.py                    # Main BBO execution pipeline
│   ├── visualize_progress.py          # Complete campaign visualization
│   └── progressive_visualize.py       # Week-by-week historical graphs
│
├── src/
│   └── bbo/                           # Core optimization modules
│       ├── __init__.py
│       ├── config.py                  # Configuration management
│       ├── data_loader.py             # Robust data loading with auto-format
│       ├── gp.py                      # Gaussian Process implementation
│       ├── io.py                      # Input/output utilities
│       ├── pipeline.py                # Main optimization pipeline
│       └── strategy.py                # Acquisition strategies
│
├── data/
│   ├── initial_data/                  # Seed data (NPY format)
│   │   └── function_1..8/
│   └── weekly/                        # Historical campaign data
│       ├── inputs.txt                 # Query history (portal format)
│       └── outputs.txt                # Evaluation results
│
├── artifacts/
│   ├── visualizations/                # Complete campaign analysis
│   ├── progressive_visualizations/    # Week-by-week historical graphs
│   │   ├── week1/                     # Week 1 cumulative view
│   │   ├── week2/                     # Week 2 cumulative view
│   │   ├── ...                        # Progressive weekly analysis
│   │   └── README.md                  # Visualization guide
│   └── submissions/                   # Portal-ready queries
│
├── docs/                             # Project documentation
│   ├── architecture.md               # System architecture (this file)
│   ├── datasheet.md                  # Dataset documentation (query history + evaluations)
│   ├── model_card.md                 # Optimisation strategy documentation
│   └── cnn_integration_guide.md      # Optional extension guide
│
├── requirements.txt
└── README.md
```

## Documentation

- [Datasheet](docs/datasheet.md)
- [Model Card](docs/model_card.md)
- [Technical details](docs/READMEAdv.md)
