# Configuration recommendations for a machine learning training program:

## Graphics Card:

Recommended graphics card is NVIDIA RTX 20XX or better. Make sure you have a graphics card from this series or even more powerful.
## Integrated Development Environment (IDE):

Recommended IDE is PyCharm. Ensure that PyCharm is installed and configured for your work.
## Dependencies Installation in the Terminal:

In the terminal, run the command pip install -r requirements.txt to install the required dependencies.
## Adjusting Configuration in config.py:

Edit the configuration in the config.py file with the following parameters:
PATH_TO_IMAGES - the path to training images
NUMBER_OF_SAMPLES_TO_LOAD - the number of samples for training
TRAINING - specifies whether to train or test the model
MODEL_INPUT_SHAPE - adjust this parameter if you want to train on larger/smaller inputs
FULLSCALE_FLOW_SHAPE - resize the output flow to this size
LOSS_WEIGHTS - weights for training the model for each output (e.g., predict_1, predict_2, etc.)
## Prediction Output:

After prediction, the data is saved in the "output" folder.
## Trained Models:

Trained models are saved in the default folder. The model named "best_model" is the one with the best validation loss. Other models with timestamps represent models after final epoch.
Ensure that you follow these instructions for a successful setup and utilization of the machine learning model training program.

## Pretrained models:
https://drive.google.com/drive/folders/1vghHe_BYpPeEs_5bK2Msjx_bSQIdmbyg