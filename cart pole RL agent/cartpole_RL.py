import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import matplotlib
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import streamlit as st
import requests
import subprocess
from collections import deque

matplotlib.use('Agg')

MODEL_DIR = "models"
VIDEO_DIR = "videos"
PPO_PATH = os.path.join(MODEL_DIR, "ppo_cartpole")
DQN_PATH = os.path.join(MODEL_DIR, "dqn_cartpole.pth")

for d in [MODEL_DIR, VIDEO_DIR]:
    os.makedirs(d, exist_ok=True)

# ==============================================================================
# DQN Implementation
# ==============================================================================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-3)
        self.buffer = ReplayBuffer(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_ts = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state_ts).argmax(dim=1).item()

    def update(self):
        if len(self.buffer) < self.batch_size: return 0
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        
        state_ts = torch.FloatTensor(state)
        next_state_ts = torch.FloatTensor(next_state)
        action_ts = torch.LongTensor(action).unsqueeze(1)
        reward_ts = torch.FloatTensor(reward).unsqueeze(1)
        done_ts = torch.FloatTensor(done).unsqueeze(1)
        
        q_values = self.policy_net(state_ts).gather(1, action_ts)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_ts).max(1)[0].unsqueeze(1)
            target_q_values = reward_ts + (1 - done_ts) * self.gamma * max_next_q_values
            
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)
        
    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path))
        self.policy_net.eval()

# ==============================================================================
# Train & Evaluate Core
# ==============================================================================
def train_dqn():
    env = gym.make("CartPole-v1")
    agent = DQNAgent(4, 2)
    episodes = 500
    rewards = []
    
    print("Training Custom PyTorch DQN...")
    for ep in range(episodes):
        state, _ = env.reset()
        ep_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.buffer.push(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            agent.update()
        
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)
        if ep % 10 == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        rewards.append(ep_reward)
        if ep % 50 == 0:
            print(f"Episode {ep} | Epsilon {agent.epsilon:.2f} | Reward: {ep_reward}")
    
    agent.save(DQN_PATH)
    env.close()
    
    plt.plot(rewards)
    plt.title("DQN Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig(os.path.join(MODEL_DIR, "dqn_learning_curve.png"))
    plt.close()
    print(f"DQN Training Complete. Saved to {DQN_PATH}")

def train_ppo():
    print("Training Stable-Baselines3 PPO...")
    env = gym.make("CartPole-v1")
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=20000)
    model.save(PPO_PATH)
    env.close()
    print(f"PPO Training Complete. Saved to {PPO_PATH}")

def run_evaluate():
    print("Evaluating models...")
    env = gym.make("CartPole-v1")
    
    # Evaluate PPO
    print("Evaluating PPO...")
    try:
        ppo_model = PPO.load(PPO_PATH)
        mean_reward, std_reward = evaluate_policy(ppo_model, env, n_eval_episodes=50)
        print(f"PPO Mean Reward: {mean_reward} +/- {std_reward}")
    except Exception:
        print("PPO Model not found.")
        
    # Evaluate DQN
    print("Evaluating DQN...")
    try:
        dqn_agent = DQNAgent(4, 2)
        dqn_agent.load(DQN_PATH)
        dqn_agent.epsilon = 0.0 # Greedy
        dqn_rewards = []
        for _ in range(50):
            state, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = dqn_agent.select_action(state)
                state, reward, term, trunc, _ = env.step(action)
                done = term or trunc
                ep_reward += reward
            dqn_rewards.append(ep_reward)
        print(f"DQN Mean Reward: {np.mean(dqn_rewards)} +/- {np.std(dqn_rewards)}")
    except Exception:
        print("DQN Model not found.")
        
    env.close()

# ==============================================================================
# FastAPI Backend
# ==============================================================================
app = FastAPI(title="CartPole Inference API")

class StateRequest(BaseModel):
    cart_position: float
    cart_velocity: float
    pole_angle: float
    pole_angular_velocity: float
    model_type: str = "ppo"

api_dqn_agent = None
api_ppo_model = None
try:
    api_ppo_model = PPO.load(PPO_PATH)
    api_dqn_agent = DQNAgent(4, 2)
    api_dqn_agent.load(DQN_PATH)
    api_dqn_agent.epsilon = 0.0
except Exception:
    pass

@app.post("/predict_action")
def predict_action(request: StateRequest):
    state = np.array([request.cart_position, request.cart_velocity, 
                      request.pole_angle, request.pole_angular_velocity])
    
    if request.model_type == "dqn":
        if not api_dqn_agent: raise HTTPException(status_code=503, detail="DQN not loaded.")
        action = api_dqn_agent.select_action(state)
    else:
        if not api_ppo_model: raise HTTPException(status_code=503, detail="PPO not loaded.")
        action, _ = api_ppo_model.predict(state, deterministic=True)
        action = int(action)
        
    return {"action": action, "action_string": "Right" if action == 1 else "Left"}

# ==============================================================================
# Streamlit & CLI Logic
# ==============================================================================
if __name__ == '__main__':
    if 'streamlit' in sys.argv[0] or (len(sys.argv) > 1 and sys.argv[1] == '--mode' and sys.argv[2] == 'ui'):
        st.title("CartPole RL Inference Sandbox")
        model_type = st.radio("Select Model", ["ppo", "dqn"])
        st.write("Input the 4 state variables below:")
        col1, col2 = st.columns(2)
        with col1:
            cp = st.number_input("Cart Position", value=0.0)
            cv = st.number_input("Cart Velocity", value=0.0)
        with col2:
            pa = st.number_input("Pole Angle", value=0.1)
            pav = st.number_input("Pole Ang. Velocity", value=0.1)
            
        if st.button("Predict Optimal Action"):
            try:
                res = requests.post("http://localhost:8000/predict_action", json={
                    "cart_position": cp, "cart_velocity": cv,
                    "pole_angle": pa, "pole_angular_velocity": pav,
                    "model_type": model_type
                })
                if res.status_code == 200:
                    st.success(f"Action: {res.json()['action_string']} ({res.json()['action']})")
                else: 
                    st.error("API error!")
            except:
                st.error("API Connection failed. Ensure the API is running on port 8000.")
                
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--mode', choices=['train_ppo', 'train_dqn', 'evaluate', 'api', 'ui'], required=True)
        args, _ = parser.parse_known_args()
        
        if args.mode == 'train_dqn': train_dqn()
        elif args.mode == 'train_ppo': train_ppo()
        elif args.mode == 'evaluate': run_evaluate()
        elif args.mode == 'api': uvicorn.run(app, host="0.0.0.0", port=8000)
        elif args.mode == 'ui': subprocess.run([sys.executable, "-m", "streamlit", "run", __file__, "--", "--mode", "ui"])
