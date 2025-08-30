
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, backend as K

class RegularizedAttention(layers.Layer):
    """Regularized attention mechanism (RAM).
    Computes per-feature, per-time weights ω_{t,m} with softmax over time for each feature m:
      ω_{t,m} = softmax_t( score_t,m )
    Reweights the input X: X_tilde[t,m] = ω_{t,m} * X[t,m]
    Adds L2 regularization on Q Q^T - I, where Q has shape (M, T).
    """
    def __init__(self, attn_units: int = 32, reg_weight: float = 1e-3, **kwargs):
        super().__init__(**kwargs)
        self.attn_units = attn_units
        self.reg_weight = reg_weight

    def build(self, input_shape):
        # input_shape: (batch, T, M)
        # store feature dimension as concrete int for later use
        self.M = int(input_shape[2])
        # temporal encoding branch
        self.conv = layers.Conv1D(
            self.attn_units, kernel_size=3, padding="same", activation="tanh"
        )
        self.proj_h = layers.Dense(self.attn_units, activation="tanh")
        self.proj_x = layers.Dense(self.attn_units, activation="tanh")
        # outputs a score for each feature at each time step
        self.score = layers.Dense(self.M)
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (B, T, M)

        # Create a simple temporal encoding via a 1D conv as a proxy for hidden features h_t
        h = self.conv(inputs)
        # Project both streams
        Hp = self.proj_h(h)                           # (B, T, U)
        Xp = self.proj_x(inputs)                      # (B, T, U)
        HXp = tf.nn.tanh(Hp + Xp)                     # (B, T, U)

        # Compute a score for every feature at each time step
        scores = self.score(HXp)                      # (B, T, M)

        # Softmax over time for each feature m independently
        scores_t = tf.transpose(scores, perm=[0, 2, 1])     # (B, M, T)
        omega = tf.nn.softmax(scores_t, axis=-1)            # (B, M, T)
        omega_t = tf.transpose(omega, perm=[0, 2, 1])       # (B, T, M)

        # Regularization: Q Q^T - I, where Q is (M, T) averaged over batch
        Q = tf.reduce_mean(omega, axis=0)                   # (M, T)
        QQT = tf.matmul(Q, Q, transpose_b=True)             # (M, M)
        I = tf.eye(self.M, dtype=QQT.dtype)
        frob = tf.reduce_sum(tf.square(QQT - I))
        self.add_loss(self.reg_weight * frob)

        # Reweight inputs
        X_tilde = inputs * omega_t
        return X_tilde, omega_t

def build_ram_bigru_model(T: int, M: int, n_classes: int, attn_units: int = 32, reg_weight: float = 1e-3, bigru_units: int = 64, dropout: float = 0.1):
    inp = layers.Input(shape=(T, M), name="input_sequence")
    x, omega = RegularizedAttention(attn_units=attn_units, reg_weight=reg_weight, name="ram")(inp)
    x = layers.Bidirectional(layers.GRU(bigru_units, return_sequences=False, dropout=dropout))(x)
    x = layers.Dense(64, activation="relu")(x)
    if n_classes == 2:
        out = layers.Dense(1, activation="sigmoid")(x)
    else:
        out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out, name="RAM_BiGRU")
    return model
