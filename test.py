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
    arr = [10,20,30,40,50,60,70,80,90]
    n_input = 3
    n_output = 1
    num_sets = 12
    n_features = 1
    X,y = create_tset(arr, n_input, n_output, num_sets)
    for i in range(len(X)):
        print(X[i],y[i])
    X = X.reshape((X.shape[0],X.shape[1],n_features))
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_input, n_features)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    model.fit(X,y,epochs=800, verbose=0)
    x_input = np.array([70,80,90])
    x_input = x_input.reshape((1,n_input,n_features))
    pred = model.predict(x_input)
    print(pred)

if __name__ == "__main__":
    main()

