import sys
import csv
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def create_univar_tset(arr, n_input, n_output, num_sets):
    X, y = [], []
    i = 0
    while(i <= len(arr) - n_input - n_output and i < num_sets):
        X.append(arr[i:i+n_input])
        y.append(arr[i+n_input:i+n_input+n_output])
        i+=1
    return np.array(X),np.array(y)

def main():
    if(len(sys.argv) != 2):
        print("Usage: train_classifier.py <file_name>")
        return
    else:
        input_file = sys.argv[1]
        print("Reading from: " + input_file)

    attr_open = []

    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        max = 0
        for row in csv_reader:
            attr_open.insert(0,float(row['open']))
        print("Done reading, closing file")

    #Define our training data
    n_input = 10
    n_output = 2
    num_sets = 100
    n_features = 1

    #Create our testing sets, and reshape for input to LSTM
    X,y = create_univar_tset(attr_open, n_input, n_output, num_sets)
    X = X.reshape((X.shape[0],X.shape[1],n_features))

    #Create the LSTM
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs = 10, verbose=1)

    #Test our prediction
    lbound = n_input*num_sets
    rbound = lbound + n_input
    x_test = np.array(attr_open[lbound:rbound])
    x_test = x_test.reshape((1,n_input,n_features))
    y_test = attr_open[rbound:rbound+n_output]
    print("Test our model:")
    print(x_test,y_test)
    y_pred = model.predict(x_test, verbose=0)
    print("Output: %f" % y_pred)


if __name__ == "__main__":
    main()
