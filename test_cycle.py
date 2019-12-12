import sys
import os
import csv
from keras.models import model_from_json
from sklearn.model_selection import StratifiedKFold
import numpy as np
from train_classifier import create_2d_tset, normalize_min_max, rse, pcc, plot_test
from sklearn.metrics import r2_score

def main():
    if(len(sys.argv) != 4):
        print("Usage: test_cycle.py <model_file_name> <weight_file_name> <data_file_name>")
        return
    else:
        model_file = sys.argv[1]
        weight_file = sys.argv[2]
        input_file = sys.argv[3]

    # Rebuild the model
    json_file = open(model_file, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights(weight_file)

    attr_open = []
    attr_volume = []
    # Load DataSet
    with open(input_file) as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            attr_open.insert(0,float(row['open']))
            attr_volume.insert(0,float(row['volume']))
    print("Done reading, closing file")

    # Normalize DataSet
    normalize_min_max(attr_open)
    normalize_min_max(attr_volume)

    n_input = 400
    n_output = 20
    num_sets = 1250
    n_features = 2

    # split into input (X) and output (Y) variables
    X,Y = create_2d_tset(attr_open, attr_volume, n_input, n_output, num_sets)

    # testing
    for i in range(1,3):
        num_sets *= i
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
        r2 = r2_score(y_pred,y_test)
        print("r2: %f" % r2)
        print("RSE: %f , PCC: %f" % (err,cc))


    quit()

if __name__ == "__main__":
    main()
