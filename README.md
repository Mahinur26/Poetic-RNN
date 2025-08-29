# Poetic RNN

This project is an LSTM-based text generator trained on Shakespeare's works. It uses TensorFlow and Keras to build and train a recurrent neural network that generates text character-by-character, with adjustable randomness via a temperature parameter.

## Features

- Loads and preprocesses Shakespeare text from TensorFlow's dataset.
- Builds a character-level LSTM model using Keras.
- Trains the model to predict the next character in a sequence.
- Generates text with adjustable creativity using temperature sampling.
- Saves and loads trained models for reuse.

## Hyperparameters

The training section in [`Main.py`](Main.py) uses the following hyperparameters:

- `SEQ_LENGTH`: Length of each input sequence (default: 40)
- `STEP_SIZE`: Step size for moving the window over the text (default: 3)
- LSTM units: 128
- Batch size: 256
- Epochs: 10
- Learning rate: 0.01 (RMSprop optimizer)

Feel free to experiment with these hyperparameters to improve model performance or adapt the training process to your needs. Adjusting sequence length, batch size, number of epochs, or the learning rate can have a significant impact on the quality and

## Usage

1. **Install dependencies:**
   ```sh
   pip install tensorflow numpy


2. **Train the model:**
   - Uncomment the training section in [`Main.py`](Main.py).
   - Run the script to train and save the model as `textgenerator.keras`.

3. **Generate text:**
   - Comment out the training section and ensure the model loading line is active.
   - Run [`Main.py`](Main.py) to generate text samples at different temperatures.

## Example Output

### 0.2 Temperature - Conservative, repetitive text sample
![0.2 Temperature Output](0.2%20temp.png)

### 1.0 Temperature - Creative, varied text sample
![1 Temperature Output](1%20temp.png)


## Files

- [`Main.py`](Main.py): Main script for training and text generation.
- `textgenerator.keras`: Saved Keras model file (created after training).

## Customization

- Change `SEQ_LENGTH` and `STEP_SIZE` in [`Main.py`](Main.py) to adjust input sequence length and training step size.
- Modify the number of LSTM units or training epochs for different model capacities.

## License

This project is for educational purposes and uses publicly available

