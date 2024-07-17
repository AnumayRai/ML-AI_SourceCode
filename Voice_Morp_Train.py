import tensorflow as tf
from tacotron2 import Tacotron2

# Assume that you have a dataset `ds` that yields (text, mel) pairs
tacotron2 = Tacotron2()

# Compile the model
tacotron2.compile(optimizer=tf.keras.optimizers.Adam(), loss=tacotron2.loss)

# Train the model
tacotron2.fit(ds, epochs=100)
from waveglow import WaveGlow

# Assume that you have a dataset `ds` that yields (mel, audio) pairs
waveglow = WaveGlow()

# Compile the model
waveglow.compile(optimizer=tf.keras.optimizers.Adam(), loss=waveglow.loss)

# Train the model
waveglow.fit(ds, epochs=100)
