#import random
import numpy as np
from collections import namedtuple, deque

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, log):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            log (Log): Reference to the log utility.
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        #return random.sample(self.memory, k=self.batch_size)
        self.rewards = np.array([e.reward for e in self.memory if e is not None]).astype(np.float32)
        self.tot = np.sum(np.exp(self.rewards))
        self.prob = np.array([np.exp(e.reward)/self.tot for e in self.memory if e is not None]).astype(np.float32) 
        selected = np.random.choice(len(self.memory), size=self.batch_size, p=self.prob)
        self.return_sample = deque(maxlen=batch_size)
        for i in selected:
            self.return_sample.append(self.memory[i])
        return self.return_sample

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)