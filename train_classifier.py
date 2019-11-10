import sys
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

#Create num_sets testing sets from arr
def create_univar_tset(arr, n_input, n_output, num_sets):
    X, y = [], []
    i = 0
    while(i <= len(arr) - n_input - n_output and i < num_sets):
        X.append(arr[i:i+n_input])
        y.append(arr[i+n_input:i+n_input+n_output])
        i+=1
    return np.array(X),np.array(y)

#Min-Max normalization: take a 1D array and linearly transform to [0,1]
def normalize_min_max(arr):
    min,max = arr[0],arr[0]
    for x in arr:
        if x > max:
            max = x
        if x < min:
            min = x

    for i in range(len(arr)):
        arr[i] = (arr[i] - min) / (max - min)

#Plot two same length data sets on the same axes
def plot_test(arr_actual, arr_predict):
    if(len(arr_actual) != len(arr_predict)):
        print("Must pass equal length arrays!")
        return
    x = np.linspace(0,1,len(arr_actual))
    fig,ax = plt.subplots()
    line1, = ax.plot(x,arr_actual, label="Actual Data")
    line2, = ax.plot(x,arr_predict, label="Predicted Data")
    ax.legend()
    plt.show()

def main():
    if(len(sys.argv) != 2):
        print("Usage: train_classifier.py <file_name>")
        return
    else:
        input_file = sys.argv[1]
        print("Reading from: " + input_file)

    attr_open = []

    #Read in stock data from the csv
    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        max = 0
        for row in csv_reader:
            attr_open.insert(0,float(row['open']))
        print("Done reading, closing file")

    #Let's normalize our input data to [0,1] via min-max
    normalize_min_max(attr_open)

    #Define our RNN
    n_nodes = 50

    #Define our training data
    n_input = 20
    n_output = 10
    num_sets = 100
    n_features = 1
    n_epochs = 1

    #Create our testing sets, and reshape for input to LSTM
    X,y = create_univar_tset(attr_open, n_input, n_output, num_sets)
    X = X.reshape((X.shape[0],X.shape[1],n_features))

    #Create the LSTM
    model = Sequential()
    model.add(LSTM(n_nodes, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs = n_epochs, verbose=1)

    #Test our prediction
    lbound = n_input*num_sets
    rbound = lbound + n_input
    x_test = np.array(attr_open[lbound:rbound])
    x_test = x_test.reshape((1,n_input,n_features))
    y_test = np.array(attr_open[rbound:rbound+n_output])
    print("Testing our model...")
    #print(x_test,y_test)
    y_pred = model.predict(x_test, verbose=0)
    #print("Output:")
    #print(y_pred)
    y_pred = y_pred.reshape(n_output)
    plot_test(y_test, y_pred)


if __name__ == "__main__":
    main()
