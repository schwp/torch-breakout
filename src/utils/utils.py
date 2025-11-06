import cv2
import numpy as np

def _clip_reward(reward:int) -> int:
    """
    Clip the reward as ask in the paper:
    - 0 if the reward is 0
    - 1 if the reward is positive
    - -1 if the reward is negative
    """
    return 0 if reward == 0 else np.sign(reward)


def _process_frame(frame):
    """
    Process the frame: grayscale, resize from 210x160 to 110x84, then cropping
    it 84x84 to keep the essential information
    """
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (84, 110))
    img = img[18:102, :]
    img = img.astype('float32') / 255.
    return img
