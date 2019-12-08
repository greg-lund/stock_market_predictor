import numpy as np
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

def create_tset(arr, n_input, n_output, num_sets):
    X,y = list(),list()
    i = 0
    while(i <= len(arr) - n_input - n_output and i < num_sets):
        X.append(arr[i:i+n_input])
        y.append(arr[i+n_input:i+n_input+n_output])
        i+=1
    return np.array(X),np.array(y)

def main():
    

if __name__ == "__main__":
    main()

