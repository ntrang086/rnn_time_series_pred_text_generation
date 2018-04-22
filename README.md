# Time Series Prediction and Text Generation using RNN

## Introduction

This project has two parts: 

* Part 1: Perform time series prediction using a Recurrent Neural Network (RNN) regressor.  In particular we will forecast the stock price of Apple 7 days in advance.
* Part 2: Create an English language sequence generator capable of building semi-coherent English sentences from scratch by building them up character-by-character. This will require a substantial amount of parameter tuning on a large training corpus (at least 100,000 characters long). In particular for this project we will be using a complete version of Sir Arthur Conan Doyle's classic book The Adventures of Sherlock Holmes.

## Code

* `RNN_project.ipynb` - Code to perform time series prediction and create a sequence generator
* `my_answers.py` - Helper code to be used in the above notebook

## Setup

* Python 3
* Install the packages in requirements.txt

### Build your Own Deep Learning Workstation

If you have access to a GPU, you should follow the Keras instructions for [running Keras on GPU](https://keras.io/getting-started/faq/#how-can-i-run-keras-on-gpu).

### Amazon Web Services

Instead of a local GPU, you could use Amazon Web Services to launch an EC2 GPU instance. (This costs money.)

## Data
All the data for the two parts are in the subdirectory `datasets`.

## Run
To run any script file, use:

`python <script.py>`

To open a notebook, use:

`jupyter notebook <notebook.ipynb>`
