from time import sleep

import gymnasium as gym
import numpy as np
import torch

from .dqn import DQN
from .utils.Buffer import Buffer
from .utils.utils import _process_frame

def play_game(env:gym.Env, filename:str, device:str):
    nb_state = env.observation_space.shape[0]
    nb_action = env.action_space.n

    model = DQN(input_dim=nb_state, output_dim=nb_action).to(device=device)
    model.load_state_dict(torch.load(filename, weights_only=True))
    model.eval()

    state, _ = env.reset()
    done = False

    frame_stack = Buffer(capacity=4)
    s_t = _process_frame(state)

    for _ in range(4):
        frame_stack.add(s_t)

    while not done:
        phi_t = np.stack(frame_stack.buffer, axis=0)

        with torch.no_grad():
            q_values = model(torch.FloatTensor(phi_t).unsqueeze(0).to(device=device))
            action = q_values.argmax().item()

        next_state, _, done, _, _ = env.step(action)

        frame_stack.add(_process_frame(next_state))

        #sleep(0.01)

    env.close()
