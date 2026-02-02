import numpy as np


class ReplayBuffer:
    """
    Optimized Replay Buffer for DDPG / MARL-DDPG.
    - Extremely fast sampling
    - Stores flattened state vectors
    - Avoids Python list overhead (pure NumPy buffer)
    """

    def __init__(self, state_dim, action_dim, max_size=200000):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        # Pre-allocate memory arrays
        self.states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards = np.zeros((max_size, 1), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim), dtype=np.float32)
        self.dones = np.zeros((max_size, 1), dtype=np.float32)

    # ---------------------------------------------------------
    def add(self, state, action, reward, next_state, done):
        """Insert transition (s, a, r, s', done)."""

        # Flatten state in case it comes as list/array of any shape
        self.states[self.ptr] = np.array(state, dtype=np.float32).reshape(-1)
        self.actions[self.ptr] = np.array(action, dtype=np.float32).reshape(-1)
        self.rewards[self.ptr] = float(reward)
        self.next_states[self.ptr] = np.array(next_state, dtype=np.float32).reshape(-1)
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    # ---------------------------------------------------------
    def sample(self, batch_size):
        """Efficient random sampling."""
        idx = np.random.choice(self.size, batch_size, replace=False)

        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_states[idx],
            self.dones[idx],
        )

    # ---------------------------------------------------------
    def __len__(self):
        return self.size
