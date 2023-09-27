import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio

capuchin_sample = os.path.join('data','Parsed_Capuchinbird_Clips','XC3776-3.wav')
not_capuchin_sample = os.path.join('data','Parsed_Not_Capuchinbird_Clips','afternoon-birds-song-in-fores-0.wav')

print(not_capuchin_sample)

def load_wav_16k_mono(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
