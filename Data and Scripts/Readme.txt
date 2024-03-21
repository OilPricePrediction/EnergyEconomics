This folder contains the scripts to generate and evaluate the Machine Learning predictions for the paper "How to Select Oil Price Prediction Models - The Effect of Statistical and Financial Performance Metrics and Sentiment Scores".

The neural network predictions are generated with the files Main.py and CoreModel.py. Specifically, running Main.py will generate results for the different prediction scenarios as specified in the scenario parameters in the Python file. 
All other predictions are generated in the "Combined Model Comparison.Rmd" file, that also serves as the evaluation and calculation of the ROI metrics.

All required / generated data files are provided in the Data/ folder.

Note: Currently only runs with Keras-Tuner version 1.0