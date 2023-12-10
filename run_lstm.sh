#!/bin/sh

# get data
wget http://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy

# run python script
python lstm_autoencoder.py
