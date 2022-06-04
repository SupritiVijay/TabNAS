# TabNAS
A Neural Network model generalized for any tabular dataset (Classification/Regression Task)

TabNAS is a generalized neural network model that can be fed any kind of tabular dataset. It has been generalized for both classification as well as regression tasks.

We have benchmarked the model for 4 datasets - two of which are classification tasks while the other two are regression. The datasets tested on are - the [Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult), [Car dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/car/), [Housing Prices dataset](https://www.kaggle.com/datasets/lespin/house-prices-dataset)

## Objective
We have generalized the function in order to automatically understand:-
* The number of Features
* Each feature type - (Categorical or Numerical)
* The number of layers and neurons required for each task

We also provide a size - performance tradeoff which ensures that the model takes up less space on the disk as well as has a good performance overall. The tradeoff between size and performance is observed to ensure the best possible values for hyperparameters can be applied.

## Input
We require the user to input the following parameters:-
* `data`: Dataset
* `target_var`: Target Variable - The variable which needs to be predicted
* `classification = True`: Classification(True/False) - Whether its a Classification task or Regression task 
* `lr`: Learning rate - The learning rate for the optimizer to train on

## Neural Architecture Search(NAS)


## Output
The project saves the model's weights which can then be loaded and worked on.


## To Execute
`python base.py`

## Size vs Performance Tradeoff Graph


