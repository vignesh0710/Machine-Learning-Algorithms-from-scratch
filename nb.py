import pandas as pd
import numpy as np
import multiprocessing
from multiprocessing import Process, Manager


def read_train_data(path):

    data = pd.read_csv(path)
    #print (data)
    return data

def training_prob_nonnumerical(data):

    global prior_prob_dict ;
    global class_conditional_prob_dict;
    set_class = set(data["class"].values.tolist())
    class_count = data["class"].value_counts()
    prior_prob_dict = {each_class: class_count[each_class]/len(data) for each_class in set_class}
    for each_class in set_class:
        filtered_data = data.loc[data["class"]==each_class]
        for each_column in filtered_data:
            if data[each_column].dtype != "float64" and data[each_column].dtype != "int64" and each_column != "class":
                set_each_column = set(filtered_data[each_column].values.tolist())
                each_column_count = filtered_data[each_column].value_counts()
                for every_label in set_each_column:
                    class_conditional_prob_dict[each_column+"^"+every_label+"^"+each_class] = each_column_count[every_label]/len(filtered_data)
    print(class_conditional_prob_dict)
    #print(prior_prob_dict)

def training_prob_numerical(data):
    pass

def predict(test_data,pred_list_multi):


    "did not multiply by the sum of class probability i.e p(y)"


    for index,rows in test_data.iterrows():
        prob_yes = [];prob_no = []
        for each_column in test_data.columns:
            #print (rows[each_column])
            if each_column+"^"+rows[each_column]+"^"+"yes" in class_conditional_prob_dict.keys() and each_column+"^"+rows[each_column]+"^"+"no" in class_conditional_prob_dict.keys():
                prob_yes.append(class_conditional_prob_dict[each_column+"^"+rows[each_column]+"^"+"yes"])
                prob_no.append(class_conditional_prob_dict[each_column +"^"+rows[each_column] +"^"+"no"])
        if np.prod(prob_yes) >= np.prod(prob_no):
            pred_list_multi.append("yes")
        else:
            pred_list_multi.append("no")
    print (pred_list_multi)




    #print (data)


if __name__ == "__main__":
    path = "/Users/vigneshsureshbabu/Desktop/nb.csv"
    test_path = "/Users/vigneshsureshbabu/Desktop/nb_test.csv"
    class_conditional_prob_dict = {};
    prior_prob_dict = {}
    s = int(len(read_train_data(test_path)))
    pred_list = []
    pred_list_multi = multiprocessing.Array('i',s)
    p = multiprocessing.Process(target=predict,args=(read_train_data(test_path),pred_list_multi))
    training_prob_nonnumerical(read_train_data(path))
    #predict(read_train_data(test_path))
    print (pred_list_multi[:])