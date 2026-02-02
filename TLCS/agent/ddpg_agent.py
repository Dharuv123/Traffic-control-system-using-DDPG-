# agent/ddpg_agent.py
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers, layers
import random

# ------------------- Actor ------------------- #
class Actor(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(400, 300)):
        super(Actor, self).__init__()
        self.fc1 = layers.Dense(hidden_units[0], activation="relu",
                                kernel_initializer=tf.keras.initializers.HeUniform())
        self.fc2 = layers.Dense(hidden_units[1], activation="relu",
                                kernel_initializer=tf.keras.initializers.HeUniform())
        self.out = layers.Dense(action_dim, activation="tanh",
                                kernel_initializer=tf.keras.initializers.RandomUniform(-0.003, 0.003))

    def call(self, state):
        x = self.fc1(state)
        x = self.fc2(x)
        return self.out(x)

# ------------------- Critic ------------------- #
class Critic(tf.keras.Model):
    def __init__(self, state_dim, action_dim, hidden_units=(400, 300)):
        super(Critic, self).__init__()
        self.fc1 = layers.Dense(hidden_units[0], activation="relu",
                                kernel_initializer=tf.keras.initializers.HeUniform())
        self.fc2 = layers.Dense(hidden_units[1], activation="relu",
                                kernel_initializer=tf.keras.initializers.HeUniform())
        self.out = layers.Dense(1, activation=None,
                                kernel_initializer=tf.keras.initializers.RandomUniform(-0.003, 0.003))

    def call(self, inputs):
        state, action = inputs
        x = self.fc1(state)
        x = tf.concat([x, action], axis=-1)
        x = self.fc2(x)
        return self.out(x)

# ------------------- Replay Buffer ------------------- #
class ReplayBuffer:
    def __init__(self, max_size=200000):
        self.buffer = []
        self.max_size = int(max_size)
        self.pos = 0

    def add(self, state, action, reward, next_state, done):
        data = (state, action, reward, next_state, done)
        if len(self.buffer) < self.max_size:
            self.buffer.append(data)
        else:
            self.buffer[self.pos] = data
        self.pos = (self.pos + 1) % self.max_size

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)

# ------------------- OU Noise ------------------- #
class OUActionNoise:
    def __init__(self, mean=0.0, std_deviation=0.1, theta=0.15, dt=1e-2, x_initial=None, size=1):
        self.theta = float(theta)
        self.mu = float(mean)
        self.sigma = float(std_deviation)
        self.dt = float(dt)
        self.size = int(size)
        self.x_prev = np.array(x_initial, dtype=np.float32) if x_initial is not None else np.zeros(self.size, dtype=np.float32)

    def __call__(self):
        noise = (self.theta * (self.mu - self.x_prev) * self.dt
                 + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.size))
        self.x_prev = self.x_prev + noise
        return self.x_prev.copy()

    def reset(self):
        self.x_prev = np.zeros(self.size, dtype=np.float32)

# ------------------- DDPG Agent ------------------- #
class DDPGAgent:
    def __init__(self, state_dim, action_dim, action_bound_low=-1.0, action_bound_high=1.0,
                 actor_lr=1e-4, critic_lr=1e-3, gamma=0.99, tau=0.005,
                 buffer_size=200000, batch_size=64, seed=None):
        if seed is not None:
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.action_bound_low = float(action_bound_low)
        self.action_bound_high = float(action_bound_high)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.batch_size = int(batch_size)

        # Networks
        self.actor = Actor(self.state_dim, self.action_dim)
        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_actor = Actor(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)

        # Build networks
        dummy_state = tf.zeros((1, self.state_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, self.action_dim), dtype=tf.float32)
        _ = self.actor(dummy_state)
        _ = self.critic([dummy_state, dummy_action])
        _ = self.target_actor(dummy_state)
        _ = self.target_critic([dummy_state, dummy_action])

        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())

        self.actor_optimizer = optimizers.Adam(learning_rate=actor_lr)
        self.critic_optimizer = optimizers.Adam(learning_rate=critic_lr)

        self.buffer = ReplayBuffer(max_size=buffer_size)
        self.noise = OUActionNoise(size=self.action_dim)

    def remember(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

    def act(self, state, explore=True):
        if isinstance(state, dict):
            actions = {}
            for key, s in state.items():
                s_arr = np.expand_dims(np.array(s, dtype=np.float32), axis=0)
                raw = self.actor(s_arr).numpy()[0]
                if explore:
                    raw = raw + self.noise()
                raw = np.clip(raw, -1.0, 1.0)
                actions[key] = self._scale_action(raw)
            return actions

        s_arr = np.expand_dims(np.array(state, dtype=np.float32), axis=0)
        raw_action = self.actor(s_arr).numpy()[0]
        if explore:
            raw_action = raw_action + self.noise()
        raw_action = np.clip(raw_action, -1.0, 1.0)
        return self._scale_action(raw_action)

    def _scale_action(self, action):
        low, high = self.action_bound_low, self.action_bound_high
        return low + (np.array(action) + 1.0) * 0.5 * (high - low)

    def train(self):
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.clip(np.array(rewards, dtype=np.float32).reshape(-1,1), -50.0, 50.0)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32).reshape(-1,1)

        next_actions = self.target_actor(next_states)
        target_q = self.target_critic([next_states, next_actions])
        y = rewards + self.gamma * (1.0 - dones) * tf.stop_gradient(target_q)

        with tf.GradientTape() as tape:
            critic_value = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(y - critic_value))
        critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
        critic_grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(critic_grads, self.critic.trainable_variables)]
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))

        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred]))
        actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
        actor_grads = [g if g is not None else tf.zeros_like(v) for g, v in zip(actor_grads, self.actor.trainable_variables)]
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))

        self.update_target(self.target_actor.variables, self.actor.variables)
        self.update_target(self.target_critic.variables, self.critic.variables)

    def update_target(self, target_weights, source_weights):
        for t, s in zip(target_weights, source_weights):
            t.assign(self.tau * s + (1.0 - self.tau) * t)

    def reset_noise(self):
        self.noise.reset()

    def save(self, base_filepath):
        base = str(base_filepath).rstrip(".h5")
        os.makedirs(os.path.dirname(base) or ".", exist_ok=True)
        self.actor.save_weights(base + "_actor.weights.h5")
        self.critic.save_weights(base + "_critic.weights.h5")
        print(f"[DDPGAgent] ✅ Saved actor -> {base}_actor.weights.h5, critic -> {base}_critic.weights.h5")

    def load(self, base_filepath):
        base = str(base_filepath).rstrip(".h5")
        self.actor.load_weights(base + "_actor.weights.h5")
        self.critic.load_weights(base + "_critic.weights.h5")
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        print(f"[DDPGAgent] ✅ Loaded actor <- {base}_actor.weights.h5, critic <- {base}_critic.weights.h5")
