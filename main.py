import os

import argparse
import ale_py
import gymnasium as gym
import torch

from src.train import train_dqn
from src.play_game import play_game


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Path to the model file')
parser.add_argument('--batch-size', type=int, default=1000, help='Number of training episodes')

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true', help='Train the agent')
group.add_argument('--load', action='store_true', help='Load the trained model')

args = parser.parse_args()

file = args.filename

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {str.upper(device)} as Torch Device for training or inference")

if args.train:
    env = gym.make("ALE/Breakout-v5")
    trained_agent = train_dqn(env=env, batch_size=args.batch_size, 
                              filename=file, device=device)
    env.close()

elif args.load:
    file = './src/models/' + file
    if not os.path.exists(file):
        print(f"Error: Model file '{args.filename}' not found.")
        exit(1)

    env = gym.make("ALE/Breakout-v5", render_mode="human")
    play_game(env=env, filename=file, device=device)
