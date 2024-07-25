# Handwritten Digit Classification

Program used to identify handwritten digits using dense and convolutional neural nets on digit image data.

Along with this readme are two files, one containing the code for creating dense neural networks 
(dense_networks.py) and the other containing code for creating convolutional neural networks (conv_networks.py). 
In the models folder, you can also find all the various models I built, which can be loaded into my programs to 
use the model on given data. These models are also the models referenced in the discussion document, which has 
been included to provide analysis of the results in much greater detail.

## Project Layout

- `data/`
	- `optdigits.tes` - testing data
	- `optdigits.tra` - training data
	- `optdigits.names` - info on how data is set up
	-  Original data can be found here: https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits
 - `code/`
	- `conv_networks.py` - program used to create 2D convolutional networks from our data
	- `dense_networks.py` - program used to create dense neural networks from our data
 - `models/` - place to store all exported dense and convolutional models
 - `results_discussion.pdf` - discussion of model hyperparameters and results

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/mctripp10/handwritten-digit-classification.git
   ```
   
2. Navigate to project directory
   ```
   cd ../handwritten-digit-classification
   ```
   
3. Install libraries
   ```bash
   pip install tensorflow
   pip install numpy
   ```

4. Configure code in the following sections within `conv_networks.py` and `dense_networks.py`:
   - PARAMETERS/HYPERPARAMETERS: change parameters/hyperparameters as desired
   - SAVE AND TEST MODEL: insert the file path of where you want each model saved, as 
well as what name you would like your model to be called

## Results
Over the course of this project, I experimented with many different parameter/hyperparameter combinations 
to find what values yielded the highest accuracies. In the end, I was able to create both dense and convolutional
networks yielding 99%+ classification accuracies on test data from `optdigits.tes`. Both of these models are stored
in `models/best/`. See `results_discussion.pdf` for further discussion on how I went about choosing these 
hyperparameters and how each model in `models/` performed. 
