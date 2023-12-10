"""
description: plots performance metrics of lstm_autoencoder
"""
import csv
import numpy
from collections import defaultdict
from matplotlib import pylab
from matplotlib import pyplot

def read_csv(csvFile):
    """
    description: reads a csvfile and returns a metrics (list of list)
    """
    with open(csvFile, 'r') as csvfile:
        listOfMetrics = defaultdict(list)
        csvReader = csv.reader(csvfile, delimiter=',')
        header = next(csvReader)
        for row in csvReader:
            for name, val in zip(header, row):
                name = name.strip(" ")
                val = val.strip(" ")
                try:
                    val = float(val)
                    listOfMetrics[name].append(val)
                except:
                    pass
    return listOfMetrics

def plot_perf(mse, valmse, acc, valacc, loss, valloss, save):
    """
    description: plots a line graph
    """
    pyplot.subplot(311)
    pylab.title("Performance metrics")
    pylab.plot(mse, label='train mse', color='orange')
    pylab.plot(valmse, label='validation mse', color='blue')
    pylab.xlabel("Epoch")
    pylab.ylabel("MSE")
    pylab.legend()

    pyplot.subplot(312)
    pylab.plot(acc, label='train accuracy', color='orange')
    pylab.plot(valacc, label='validation accuracy', color='blue')
    pylab.xlabel("Epoch")
    pylab.ylabel("Accuracy")
    pylab.legend()

    pyplot.subplot(313)
    pylab.plot(loss, label='train loss', color='orange')
    pylab.plot(valloss, label='validation loss', color='blue')
    pylab.xlabel("Epoch")
    pylab.ylabel("Loss")
    pylab.legend()
    pyplot.savefig(save)
    pylab.show()

def plot_perf1(mse,  save):
    """
    description: plots a line graph
    """
    pylab.title("Performance metrics")
    pylab.plot(mse, label='train mse', color='orange')
    pylab.plot(valmse, label='validation mse', color='blue')
    pylab.xlabel("Epoch")
    pylab.ylabel("MSE")
    pylab.legend()
    pyplot.savefig(save)
    pylab.show()

def plot_performance_reconstruction():
    """
    function to plot performance of reconstruction lstm autoencoder
    """
    csvFile = 'epoch_100_sample_1000_rec.csv'
    listOfMetrics = read_csv(csvFile)
    mse = listOfMetrics['mse']
    acc = listOfMetrics['accuracy']
    loss = listOfMetrics['loss']
    valmse = listOfMetrics['val_mse']
    valacc = listOfMetrics['val_accuracy']
    valloss = listOfMetrics['val_loss']
    saveFile = 'performance_reconstruction.png'
    plot_perf(mse, valmse, acc, valacc, loss, valloss, saveFile)

def plot_performance_prediction():
    """
    function to plot performance of prediction lstm autoencoder
    """
    csvFile = 'epoch_100_sample_1000_pred.csv'
    listOfMetrics = read_csv(csvFile)
    #mse = listOfMetrics['mse']
    acc = listOfMetrics['accuracy']

    #loss = listOfMetrics['loss']
    #valmse = listOfMetrics['val_mse']
    #valacc = listOfMetrics['val_accuracy']
    #valloss = listOfMetrics['val_loss']
    saveFile = 'performance_prediction.png'
    plot_perf(mse, valmse, acc, valacc, loss, valloss, saveFile)

def main():
    plot_performance_reconstruction()
    plot_performance_prediction()

if __name__=="__main__":
    main()
