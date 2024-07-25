# Handwritten Digit Recognition

Program used to identify handwritten digits using dense and convolutional neural nets on digit image data.

![handwritten-digit-img](https://miro.medium.com/v2/resize:fit:720/format:webp/1*SfRJNb5dOOPZYEFY5jDRqA.png)
###### Image credit: [Koushik](https://medium.com/@koushikkushal95/mnist-hand-written-digit-classification-using-neural-network-from-scratch-54da85712a06)

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
   git clone https://github.com/mctripp10/handwritten-digit-recognition.git
   ```
   
2. Navigate to project directory
   ```
   cd ../handwritten-digit-recognition
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

## Loading/Saving Models

### Saving
Currently, the model currently defined in the program will automatically be saved on run according to the path and file name defined in the "Save and Test Model" section. 
Comment out this code if you do not want it to save.
```Python
modelName = "model_name.h5"
filepath = "./models/model_type/"       # Insert model file path here
model.save(f"{filepath}{modelName}")
```
 ### Loading
 Currently, the model currently defined in the program will also automatically be loaded and tested according to the path and file name defined in the "Save and Test Model"
 section. If you wish to load a specific model you have already saved, simply replace the `model_to_load` assignment with your file path:
 ```Python
model_to_load = "insert/your/file/path/here"          # File path for desired model to load and test
```

## Results
Over the course of this project, I experimented with many different parameter/hyperparameter combinations 
to find what values yielded the highest accuracies. In the end, I was able to create both dense and convolutional
networks yielding 99%+ classification accuracies on test data from `optdigits.tes`. Both of these models are stored
in `models/best/`. See `results_discussion.pdf` for further discussion on how I went about choosing these 
hyperparameters and how each model in `models/` performed. 
