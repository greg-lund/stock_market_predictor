import sys
import os
import csv
import numpy as np
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import LSTM
from keras.layers import Dense
import matplotlib.pyplot as plt

#Create num_sets testing sets from arr
def create_univar_tset(arr, n_input, n_output, num_sets):
    if(num_sets == "all"):
        num_sets = np.Inf
    X, y = [], []
    i = 0
    while(i <= len(arr) - n_input - n_output and i < num_sets):
        X.append(arr[i:i+n_input])
        y.append(arr[i+n_input:i+n_input+n_output])
        i+=1
    return np.array(X),np.array(y)

#Create a testing set from arr1 and arr2
#arr1 will be used for output
def create_2d_tset(arr1, arr2, n_input, n_output, num_sets):
    if(num_sets == "all"):
        num_sets = np.Inf
    if(len(arr1) < len(arr2)):
        l = len(arr1)
    else:
        l = len(arr2)

    arr = np.stack((arr1[0:l],arr2[0:l]), axis = -1)
    X, y = [], []
    i = 0

    while(i <= l - n_input - n_output and i < num_sets):
        X = np.append(X, arr[i:i+n_input])
        y = np.append(y, arr1[i+n_input:i+n_input+n_output])
        i+=1

    return X.reshape(i,n_input,2),y.reshape(i,n_output)


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
def plot_test(arr_input, arr_actual, arr_predict):
    if(len(arr_actual) != len(arr_predict)):
        print("Must pass equal length arrays!")
        return
    x = np.linspace(0,1,len(arr_actual))
    plt.subplot(2,1,1)
    plt.plot(arr_input)
    plt.title('Input data')
    plt.subplot(2,1,2)
    plt.plot(x,arr_actual,x,arr_predict)
    plt.gca().legend(('Actual','Predicted'))
    plt.title('Output and Actual')
    plt.show()

#Relative squared error
def rse(x, y):
    n = len(x)
    err = 0.0
    for i in range(0,n):
        a = (x[i] - y[i]) / y[i]
        err += a**2
    err /= n
    return err

#Pearson Correlation Coefficient
def pcc(x,y):
    cc,mean1,mean2,prod_sum,stdev1,stdev2 = 0,0,0,0,0,0
    n = len(x)
    for i in range(0,n):
        mean1 += x[i]
        mean2 += y[i]
        prod_sum += x[i]*y[i]
    mean1 /= n
    mean2 /= n
    for i in range(0,n):
        stdev1 += (x[i] - mean1)**2
        stdev2 += (y[i] - mean2)**2
    stdev1 = np.sqrt(stdev1/n)
    stdev2 = np.sqrt(stdev2/n)

    cc = (prod_sum - n * mean1 * mean2) / (n * stdev1 * stdev2)
    return cc

def main():
    if(len(sys.argv) != 3):
        print("Usage: train_classifier.py <data_file_name> <output_model_name>")
        return
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        print("Reading from: " + input_file)

    attr_open = []
    attr_volume = []
    #Read in stock data from the csv
    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            attr_open.insert(0,float(row['open']))
            attr_volume.insert(0,float(row['volume']))
    print("Done reading, closing file")

    #Let's normalize our input data to [0,1] via min-max
    normalize_min_max(attr_open)
    normalize_min_max(attr_volume)

    #Define our RNN
    nodes = [500,500,400,400,300,200]

    #define our training data
    n_input = 800
    n_output = 10
    num_sets = 3200
    n_features = 2
    n_epochs = 40

    #create our testing sets, and reshape for input to lstm
    X,y = create_2d_tset(attr_open, attr_volume, n_input, n_output, num_sets)

    #create the lstm
    model = Sequential()
    model.add(LSTM(nodes[0], activation='relu', input_shape=(n_input, n_features)))
    for i in range(1,len(nodes)):
        model.add(Dense(nodes[i]))
    model.add(Dense(n_output))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs = n_epochs, verbose=1)

    #Write our model to our output file name
    model_out = model.to_json()
    with open(output_file + ".json", "w") as ofile:
        ofile.write(model_out)
    model.save_weights(output_file + ".h5")
    print("LSTM saved as %s, nodes saved as %s" % ((output_file + ".json"), (output_file + ".h5")))


    #Test our prediction
    lbound = num_sets + 1
    rbound = lbound + n_input
    x_open = attr_open[lbound:rbound]
    x_volume = attr_volume[lbound:rbound]
    x_test = np.stack((x_open,x_volume), axis=-1)
    x_test = x_test.reshape((1,n_input,n_features))

    y_test = np.array(attr_open[rbound:rbound+n_output])
    print("Testing our model...")
    y_pred = model.predict(x_test, verbose=0)
    y_pred = y_pred.reshape(n_output)
    err = rse(y_pred, y_test)
    cc = pcc(y_pred, y_test)
    print("RSE: %f , PCC: %f" % (err,cc))
    plot_test(x_open,y_test, y_pred)


if __name__ == "__main__":
    main()
