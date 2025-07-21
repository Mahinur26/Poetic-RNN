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
    sentences.append(text[i: i + SEQ_LENGTH])
    next_characters.append(text[i + SEQ_LENGTH])

#Input Layer
#3D array with number of sentences, SEQ_LENGTH, and number of unique characters
#If a character is present in the sentence, it is set to True/1
x = np.zeros((len(sentences), SEQ_LENGTH, len(characters)), dtype=bool)

#Contains the next character for each sentence
#Acts as correct output for the model
y = np.zeros((len(sentences), len(characters)), dtype=bool)

#Assigning an indecx to each sentance with the enumerate function
for i, sentance in enumerate(sentences):
    #Also enumerate each character in the sentence
    for t, character in enumerate(sentance):
        #Set the index for the character in the sentence to True/1 because it actually is there
        x[i, t, character_to_index[character]] = 1
    #Set the next character for the sentence to True/1
    y[i, character_to_index[next_characters[i]]] = 1


##Use this section to train the model and once done with training, comment it out or delete it
#Creating the model now that the data is ready
model = Sequential()

#Adding an LSTM layer with 128 units
#Input is immediatly put into the LSTM layer
model.add(LSTM(128, input_shape=(SEQ_LENGTH, len(characters))))

#The LSTM cells are followed by a Dense/hidden layer
model.add(Dense(len(characters)))

#The activation function is softmax to get a probability distribution over the characters
#Softmax makes all the vectors/probabilities output sum to 1
model.add(Activation("softmax"))

#Calculating the loss with categorical crossentropy
#Using RMSprop as the optimizer with a learning rate of 0.01 to begin backpropagation
model.compile(loss="categorical_crossentropy", optimizer=RMSprop(learning_rate=0.01))

#Backpropagation through time (BPTT) to train the model
#The hyperparamters - batch size, epochs, and learning rate can be adjusted
#Batch size is the number of training examples used in one iteration
#Epochs is the number of times the model will see the same training examples again
model.fit(x,y,batch_size=256, epochs=10)

# Save after training
model.save("textgenerator.keras")

# Then load here
model = tf.keras.models.load_model("textgenerator.keras")

#End of training section

'''
#Helper function to generate text (Copied from the keras tutorial)
#Takes the predictions from the model and chooses the next character based on a temperature parameter
#The temperature parameter is used to decide if the next character should be more conservative or more random/experimental
#Higher temperature means more random choices, lower temperature means more conservative choices
def sample(preds, temperature=1.0):
    #Sampling from the probability distribution
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def generate_text(length, temperature):
    #The first 40 characters(SEQ_LENGTH) are used as the starting point/context and are from the dataset (Randomly chosen)
    #After that, the model will generate the rest of the text
    start_index = np.random.randint(0, len(text) - SEQ_LENGTH -1)
    #Will contain the entirity of the random sequence + the generated text
    generated_text = ""
    sentence = text[start_index: start_index + SEQ_LENGTH]
    generated_text += sentence

    for i in range(length):
        #Input array of shape: 1 for the batch size, SEQ_LENGTH for the number of timesteps, len(characters) for one-hot encoding size
        x = np.zeros((1, SEQ_LENGTH, len(characters)))
        for t, character in enumerate(sentence):
            #Set the index for the character in the sentence to True/1 because it actually is there
            x[0, t, character_to_index[character]] = 1
            
        #Get the predictions from the model (The softmax output)
        predictions = model.predict(x, verbose=0)[0]
        #Temperature sampling to choose the next character
        #Adjust the temperature to make the model more or less random  
        next_index = sample(predictions, temperature)
        #Converts the index back to the character
        next_character = index_to_character[next_index]
        #Appends the next character to the generated text
        generated_text += next_character
        #Slides the input sentence by one character
        #This will be used as the input for the next prediction
        sentence = sentence[1:] + next_character    
    return generated_text
print("--------------0.2 Temperature----------------")
print(generate_text(300, 0.2))
print("--------------0.6 Temperature----------------")
print(generate_text(300, 0.6))
print("--------------0.1 Temperature----------------")
print(generate_text(300, 0.1))
print("--------------0.8 Temperature----------------")
print(generate_text(300, 0.8))
'''