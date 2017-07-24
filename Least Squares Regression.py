import pandas as pd
import numpy as np
from math import sqrt



def read_input(path):

    """
    reads the input csv file and returns a df of the dataset
    :return:
    """
    data = pd.read_csv(path)
    for each_column in data.columns:
        if each_column != "class":
            mean = data[each_column].mean(); std = data[each_column].std()
            data[each_column] = (data[each_column]-mean)/std

    data['intercept'] = np.ones((len(data)))

    return data

def hypothesis_calculator(data):

    global weight_vector
    print ("calculating with weight_vector: ",weight_vector)
    hyp_list = [];
    column_list = ["area",'beds','intercept']
    #column_list = data.columns.tolist().remove("class")
    actual_values = data["class"].values.tolist()
    for index,rows in data.iterrows():
        temp = []
        for i in range(len(column_list)):
            temp.append(rows[column_list[i]]*weight_vector[i])
        hyp_list.append(sum(temp))
    return [a_i - b_i for a_i, b_i in zip(hyp_list, actual_values)]



def fit(data):
    global weight_vector
    global tolerance
    column_list = ["area",'beds','intercept']
    #column_list = data.columns.tolist()
    while tolerance > accepted_tolerance:
        temp_weight_vector = []
        weighted_error = 0
        summation_list = hypothesis_calculator(data)
        for i  in range(len(weight_vector)):
            respective_feature_value = data[column_list[i]].tolist()
            weighted_error = sum([a * b for a, b in zip(summation_list, respective_feature_value)])
            weight_vector[i] = weight_vector[i] - (learning_rate*weighted_error)
            temp_weight_vector.append(weight_vector[i])


        weight_vector = temp_weight_vector
        #print(sum(summation_list))
        tolerance = sqrt((sum(summation_list)**2) + weighted_error**2)
        print ("weight_vector----> ",weight_vector)
        print ("tolerance  ",tolerance)

def predict(test_path):

    global weight_vector
    test_data =  pd.read_csv(test_path)
    print (test_data)
    pred_list = []
    column_list = [test_data.columns]
    for index,rows in data.iterrows():
        temp = []
        for i in range(len(column_list)):
            temp.append(rows[column_list[i]]*weight_vector[i])
        pred_list.append(sum(temp))
    print (pred_list)






if __name__ == '__main__':
    path = "/Users/vigneshsureshbabu/Desktop/cars.csv"
    test_path = "/Users/vigneshsureshbabu/Desktop/cars_test.csv"

    learning_rate = 0.1
    data = read_input(path)
    weight_vector = list(np.zeros(len(data.columns)-1))
    tolerance = 9999999999
    accepted_tolerance = 0.1
    fit(data)
    predict(test_path)



