from pathlib import Path

ACQUISITION="ei"  # "ei", "pi", "ucb"
N_CANDIDATES=70000
RNG_SEED=123
DECIMALS=6

NOISE_ALPHA=1e-10
RESTARTS_LOW_D=10
RESTARTS_MID_D=6
RESTARTS_HIGH_D=2

LENGTH_SCALE_BOUNDS=(1e-3, 50.0)
WHITE_NOISE_BOUNDS=(1e-12, 1e-1)

EXPLOIT_TOL_FRAC_OF_RANGE=0.10
XI_EXPLOIT=0.001
XI_EXPLORE=0.05
BETA_EXPLOIT=1.0
BETA_EXPLORE=3.0

# Week 6: Advanced Optimization Parameters
ADAPTIVE_EXPLORATION_RATE=0.85  # Decay rate for exploration over time
CONVERGENCE_WINDOW=3  # Window size for convergence analysis
CONVERGENCE_THRESHOLD=1e-4  # Improvement threshold for convergence
MULTI_OBJECTIVE_WEIGHTS=[1.0, 0.8, 0.6, 0.4, 1.2, 1.5, 0.9, 1.1]  # Function portfolio balancing
UNCERTAINTY_BOOST_FACTOR=1.5  # Enhanced uncertainty quantification multiplier
PARAMETER_ADAPTATION_RATE=0.1  # Rate for sophisticated parameter tuning
MIN_IMPROVEMENT_THRESHOLD=1e-5  # Minimum improvement for mode switching

HISTORY_DIR=Path("history")
ARTIFACTS_DIR=Path("artifacts")
SUBMISSIONS_DIR=ARTIFACTS_DIR/"submissions"
