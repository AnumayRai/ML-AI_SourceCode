import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define the encoder
class Encoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.enc_units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state = hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

# Define the decoder
class Decoder(keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.gru = layers.GRU(self.dec_units,
                               return_sequences=True,
                               return_state=True,
                               recurrent_initializer='glorot_uniform')
        self.fc = layers.Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)
        x = tf.reshape(x, (-1, x.shape[1], x.shape[2]))
        output, state = self.gru(x, initial_state = hidden)
        output = tf.reshape(output, (-1, output.shape[1] * output.shape[2]))
        x = self.fc(output)
        return x, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.dec_units))

# Define the sequence-to-sequence model
def Seq2Seq(vocab_size, embedding_dim, enc_units, dec_units, batch_sz):
    encoder = Encoder(vocab_size, embedding_dim, enc_units, batch_sz)
    decoder = Decoder(vocab_size, embedding_dim, dec_units, batch_sz)

    def call(self, inputs, states=None):
        enc_output, enc_state = encoder(inputs, states)
        dec_state = enc_state
        loss = 0
        for i in range(1, targ_length):
            # passing enc_output to the decoder as its first input
            dec_output, dec_state = decoder(enc_output, dec_state)
            # calculate the loss for each step
            loss += self.compiled_loss(y_true[:,i], dec_output, regularization_losses=self.losses)
        return loss

    model = keras.Model(inputs=inputs, outputs=call)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
    return model
