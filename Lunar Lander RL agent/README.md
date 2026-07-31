# Lunar Lander RL Agent

## Overview
This project implements deep reinforcement learning agents to solve the Gymnasium LunarLander-v3 environment. Agents learn to fire engines precisely to land safely on the lunar surface.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Train agents using `python lunar_lander_pipeline.py --mode train_dqn` or `--mode train_ppo`
3. Evaluate trained models with `python lunar_lander_pipeline.py --mode evaluate`
4. Launch API or UI interfaces as needed

## Directory Structure
- `models/`: Saved trained models and checkpoints
- `scripts/`: Training and evaluation scripts
- `videos/`: Rendered gameplay recordings
- `src/`: Source code for environment interaction and models

## Usage
Execute training commands sequentially. Use evaluation mode to assess model performance on standardized test episodes.

## Contributing
Contributions are welcome. Follow established coding patterns and submit pull requests for enhancements.

## License
MIT License