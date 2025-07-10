#LSTM Text Generation with TensorFlow
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Activation
from tensorflow.keras.optimizers import RMSprop

filepath = tf.keras.utils.get_file("shakespeare.txt", "https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt")

#Uppercase text adds too many unique characters which would lead to worse performance
#The text is read in binary mode then decoded with utf-8 encoding
text = open(filepath, "rb").read().decode(encoding="utf-8").lower()

text = text[200000:800000] # We take a subset of the text to speed up training

#All unique characters are filtered and sorted
#Most likely will be the whole alphabet, space, and some punctuation
characters = sorted(set(text))

#Create a mapping from characters to indices and vice versa
#Will be used to convert characters to numbers and back
character_to_index = dict((c, i) for i, c in enumerate(characters))
index_to_character = dict((i, c) for i, c in enumerate(characters))

SEQ_LENGTH = 40
#Used to incrament the starting point of the next sequence
STEP_SIZE = 3

#The indicies for each sentence and next character are parallel
#Contains each of the training examples
sentences = []
#Contains the correct next character for each training example
next_characters = []

#the -SEQ_LENGTH ensures that each incramnet has a full sequence
for i in range(0, len(text) - SEQ_LENGTH, STEP_SIZE):
    sentences.append(text[i: i+SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

#Input Layer
x = np.zeros(len(sentences), SEQ_LENGTH, len(characters), dtype=bool)
#Output Layer
y = np.zeros(len(sentences), len(characters), dtype=bool)