# Stock Market Prediction

## About this Project
This project served as the final project for CSCI: 4502, Data Mining. The goal of the project is to create a classifier, utilizing a LSTM neural network, to accurately predict future stock market prices. The data is sourced from AlphaVantage's API, and is preprocessed and fed into a Keras LSTM. Results are shown below.

## Completed Testing Runs
### Initial Multivariate Testing
This test utilized a network with an input data stream of daily stock price and volume sold going 80 weeks into the past, with a prediction of 4 weeks into the future. The network consists of 4 hidden layers of size 400,400,300 and 200 nodes. It was trained on 1250 sets. 

![Initial 2D Results](/graphs/2d_test_1.png) <!-- .element width="50%" -->
