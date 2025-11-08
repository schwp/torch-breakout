import random

import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

from .dqn import DQN
from .utils.Buffer import Buffer
from .utils.utils import _clip_reward, _process_frame


def train_dqn(env:gym.Env, batch_size:int=32, total_frames:int = 10000000, 
              epsilon:float=1.0, min_epsilon:float=0.1, k:int=4, gamma:float = 0.99, 
              filename:str | None = None, device:str | None = None):
    decay_frames = np.max(total_frames // 10, 1000000)
    episode = 0

    nb_state = env.observation_space.shape[0]
    nb_action = env.action_space.n
    model = DQN(input_dim=nb_state, output_dim=nb_action).to(device=device)

    epsilon_decay = (epsilon - min_epsilon) / decay_frames
    optimizer = optim.RMSprop(model.parameters(), lr=0.00025, alpha=0.95, eps=1e-2)
    criterion = nn.MSELoss()

    buffer = Buffer(capicity=100000)

    frame = 0
    eps = epsilon

    # Loop for the all number of frame (here 10M)
    while frame < total_frames:
        # Init the environnement
        state, _ = env.reset()
        done = False
        frame_stack = Buffer(capicity=4)
        s_t = _process_frame(state)

        for _ in range(4):
            frame_stack.add(s_t)

        # Execute a whole game and save frames in memory
        while not done and frame < total_frames:
            phi_t = np.stack(frame_stack.buffer, axis=0)

            # Apply the epsilon-greedy
            if random.random() < eps:
                a_t = env.action_space.sample()
            else:
                with torch.no_grad():
                    q_values = model(torch.FloatTensor(phi_t).unsqueeze(0).to(device=device))
                    a_t = q_values.argmax().item()

            # Step in the environment and save the new state
            s_t1, r, done, _, _ = env.step(a_t)

            r = _clip_reward(r)
            frame += 1

            s_t1 = _process_frame(s_t1)
            frame_stack.add(s_t1)
            phi_t1 = np.stack(frame_stack.buffer, axis=0)

            buffer.add((phi_t, a_t, r, phi_t1, done))

            # If we have enough data to update and this is the right time to
            # update our weights 
            if len(buffer) >= batch_size and frame % k == 0:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)

                states = torch.FloatTensor(states).to(device=device)
                actions = torch.LongTensor(actions).to(device=device)
                rewards = torch.FloatTensor(rewards).to(device=device)
                next_states = torch.FloatTensor(next_states).to(device=device)
                dones = torch.FloatTensor(dones).to(device=device)

                q = model(states).gather(1, actions.unsqueeze(1)).squeeze()

                with torch.no_grad():
                    pred = model(next_states).max(1)[0]
                    y_hat = rewards + gamma * pred * (1 - dones)

                loss = criterion(q, y_hat)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10)
                optimizer.step()

            # Epsilon update
            eps = max(min_epsilon, eps - epsilon_decay)

        episode += 1
        if episode % 100 == 0:
            print(f"Episode {episode}, Frames: {frame:,}, Epsilon: {eps:.3f}")

    file_path = f'./src/models/{filename if filename is not None else "default.pth"}'
    torch.save(model.state_dict(), file_path)
    print(f'Your model has been saved in "{file_path}"')
