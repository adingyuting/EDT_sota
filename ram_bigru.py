
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
        T = input_shape[1]
        M = input_shape[2]
        # Precreate temporal convolution and projection layers to avoid
        # dynamic variable creation inside `call`, which is disallowed when
        # the layer is wrapped with `tf.function`.
        self.temporal_conv = layers.Conv1D(
            self.attn_units, kernel_size=3, padding="same", activation="tanh"
        )
        self.proj_h = layers.Dense(self.attn_units, activation="tanh")
        self.proj_x = layers.Dense(self.attn_units, activation="tanh")
        self.score = layers.Dense(1)  # scalar per time-feature after broadcasting
        super().build(input_shape)

    def call(self, inputs, training=None):
        # inputs: (B, T, M)
        B = tf.shape(inputs)[0]
        T = inputs.shape[1]
        M = inputs.shape[2]

        # Create a simple temporal encoding via a 1D conv as a proxy for hidden
        # features h_t. The convolutional layer is created in `build` and reused
        # for every call.
        h = self.temporal_conv(inputs)
        # Project both streams
        Hp = self.proj_h(h)                           # (B, T, U)
        Xp = self.proj_x(inputs)                      # (B, T, U)
        HXp = tf.nn.tanh(Hp + Xp)                     # (B, T, U)

        # Compute scores for each (t, m) by broadcasting along feature dim
        # Expand feature dimension: tile dense across M using time-distributed approach
        scores = self.score(HXp)                      # (B, T, 1)
        scores = tf.tile(scores, [1, 1, M])           # (B, T, M)

        # Softmax over time for each feature m independently
        scores_t = tf.transpose(scores, perm=[0, 2, 1])     # (B, M, T)
        omega = tf.nn.softmax(scores_t, axis=-1)            # (B, M, T)
        omega_t = tf.transpose(omega, perm=[0, 2, 1])       # (B, T, M)

        # Regularization: Q Q^T - I, where Q is (M, T) averaged over batch
        Q = tf.reduce_mean(omega, axis=0)                   # (M, T)
        QQT = tf.matmul(Q, Q, transpose_b=True)             # (M, M)
        I = tf.eye(M, dtype=QQT.dtype)
        frob = tf.norm(QQT - I, ord="fro", axis=[-2, -1])
        self.add_loss(self.reg_weight * frob)

        # Reweight inputs
        X_tilde = inputs * omega_t
        return X_tilde, omega_t

def build_ram_bigru_model(T: int, M: int, n_classes: int, attn_units: int = 32, reg_weight: float = 1e-3, bigru_units: int = 64, dropout: float = 0.1):
    inp = layers.Input(shape=(T, M), name="input_sequence")
    x, omega = RegularizedAttention(attn_units=attn_units, reg_weight=reg_weight, name="ram")(inp)
    x = layers.Bidirectional(layers.GRU(bigru_units, return_sequences=False, dropout=dropout))(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs=inp, outputs=out, name="RAM_BiGRU")
    return model
