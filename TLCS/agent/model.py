import tensorflow as tf
from tensorflow.keras import layers, initializers


# ================================================================= #
#                          ACTOR NETWORK                            #
# ================================================================= #
class Actor(tf.keras.Model):
    """
    Highly optimized Actor for MARL-DDPG.
    - LayerNorm after linear layers
    - He-normal initialization
    - Smooth activation (ReLU)
    - Stable tanh output
    """

    def __init__(self, state_dim, action_dim, hidden_units=(256, 256)):
        super().__init__()

        # Layer 1
        self.fc1 = layers.Dense(
            hidden_units[0],
            activation=None,
            kernel_initializer=initializers.he_normal()
        )
        self.ln1 = layers.LayerNormalization()
        self.act1 = layers.ReLU()

        # Layer 2
        self.fc2 = layers.Dense(
            hidden_units[1],
            activation=None,
            kernel_initializer=initializers.he_normal()
        )
        self.ln2 = layers.LayerNormalization()
        self.act2 = layers.ReLU()

        # Output action ∈ [-1,1]
        self.out = layers.Dense(
            action_dim,
            activation="tanh",
            kernel_initializer=initializers.RandomUniform(minval=-0.003, maxval=0.003)
        )

    def call(self, state):
        x = self.fc1(state)
        x = self.ln1(x)
        x = self.act1(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)

        return self.out(x)


# ================================================================= #
#                          CRITIC NETWORK                           #
# ================================================================= #
class Critic(tf.keras.Model):
    """
    Optimized Critic for MARL-DDPG.
    - State → LayerNorm → ReLU
    - Then concat Action
    - Stability LayerNorm on second dense
    - Small output initialization
    """

    def __init__(self, state_dim, action_dim, hidden_units=(256, 256)):
        super().__init__()

        # First layer: only state
        self.fc1 = layers.Dense(
            hidden_units[0],
            activation=None,
            kernel_initializer=initializers.he_normal()
        )
        self.ln1 = layers.LayerNormalization()
        self.act1 = layers.ReLU()

        # Second layer: state ⊕ action
        self.fc2 = layers.Dense(
            hidden_units[1],
            activation=None,
            kernel_initializer=initializers.he_normal()
        )
        self.ln2 = layers.LayerNormalization()
        self.act2 = layers.ReLU()

        # Output Q value
        self.out = layers.Dense(
            1,
            activation=None,
            kernel_initializer=initializers.RandomUniform(minval=-0.003, maxval=0.003)
        )

    def call(self, inputs):
        state, action = inputs

        x = self.fc1(state)
        x = self.ln1(x)
        x = self.act1(x)

        # ACTION CONCATENATION
        x = tf.concat([x, action], axis=-1)

        x = self.fc2(x)
        x = self.ln2(x)
        x = self.act2(x)

        return self.out(x)
