# Cart Pole RL Agent

## Overview
This project implements deep reinforcement learning agents to solve the Gymnasium CartPole-v1 environment. Two approaches are provided: a custom PyTorch DQN agent and a PPO agent using Stable-Baselines3, both encapsulated in a single pipeline file.

## Quick Start
1. Install dependencies: `pip install -r requirements.txt`
2. Train DQN agent: `python cartpole_pipeline.py --mode train_dqn`
3. Train PPO agent: `python cartpole_pipeline.py --mode train_ppo`
4. Evaluate models: `python cartpole_pipeline.py --mode evaluate`
5. Launch API: `python cartpole_pipeline.py --mode api`
6. Start Streamlit UI: `python cartpole_pipeline.py --mode ui`

## Directory Structure
- `models/`: Saved trained models and checkpoints
- `scripts/`: Training and evaluation scripts
- `videos/`: Rendered gameplay recordings
- `src/`: Core algorithm implementations

## Usage
Execute training commands sequentially. Use evaluation to validate performance, and deploy via API or UI interfaces as needed.

## Contributing
Contributions are welcome. Follow established coding patterns and submit enhancements through pull requests.

## License
MIT License