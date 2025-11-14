import os

import argparse
import ale_py
import gymnasium as gym
import torch

from src.train import train_dqn
from src.play_game import play_game


parser = argparse.ArgumentParser()
parser.add_argument('filename', type=str, help='Path to the model file')
parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
parser.add_argument('--frames', type=int, default=10000000, help='Number of training total frames')
parser.add_argument('--game', type=str, default="breakout", 
                    choices=["breakout", "pong"],
                    help="Game to train/play a model on (Breakout or Pong)")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('--train', action='store_true', help='Train the agent')
group.add_argument('--load', action='store_true', help='Load the trained model')

args = parser.parse_args()

file = args.filename

game = "ALE/Breakout-v5" if args.game == "breakout" else "ALE/Pong-v5"

if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

print(f"Using {str.upper(device)} as Torch Device for training or inference")

if args.train:
    env = gym.make(game)
    trained_agent = train_dqn(env=env, batch_size=args.batch_size, 
                              total_frames=args.frames, filename=file, device=device)
    env.close()

elif args.load:
    file = './src/models/' + file
    if not os.path.exists(file):
        print(f"Error: Model file '{args.filename}' not found.")
        exit(1)

    env = gym.make(game, render_mode="human", frameskip=1)
    play_game(env=env, filename=file, device=device)
