# Handwritten Digit Classification

Program used to identify handwritten digits using dense and convolutional neural nets on digit image data.

Along with this readme are two files, one containing the code for creating dense neural networks 
(dense_networks.py) and the other containing code for creating convolutional neural networks (conv_networks.py). 
In the models folder, you can also find all the various models I built, which can be loaded into my programs to 
use the model on given data. These models are also the models referenced in the discussion document, which has 
been included to provide analysis of the results in much greater detail.

### Project Layout

Within each file, the sections are divided by subject headings:

Under setup, you will need to insert the file path of where your training and testing data files are. See Data 
folder to use the data that I used:
  - `optdigits.names` - info on how data is set up
  - `optdigits.tes` - testing data
  - `optdigits.tra` - training data
	
PARAMETERS/HYPERPARAMETERS: Can change parameters/hyperparameters in this section (surprise).

SAVE AND TEST MODEL: under this section, will need to insert the file path of where you want each model saved, as 
well as what name you would like your model to be called.
