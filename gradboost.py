import pandas as pd
import math
from collections import Counter
import operator

def sigmoid(x):
    #the sigmoid function used for finding the regression value
  return  1 / (1 + math.exp(-x))


def zero_branch_variance_calculation(df):

    zero_branch_mean_dict = {}
    zero_branch_weighted_mean_dict = {}
    for each_column in df.columns:

        if each_column != 'regression_value' and each_column != 'gradient' and each_column != 'sigmoid_reg_value' and each_column != 'class':
        #if each_column == 'stalk-color-above-ring-buff':
            if len(list(set(df[each_column]))) == 2 or (len(list(set(df[each_column]))) == 1 and list(set(df[each_column]))[0] == 0):
                left_df = df[[each_column, "gradient"]][df[each_column] == 0]
                #print (each_column)
                #print ("mean",left_df['gradient'].mean())
                #print ("weighted_mean",(len(left_df)/len(df))* left_df['gradient'].mean())
                zero_branch_mean_dict[each_column] = round(left_df['gradient'].mean(),4)
                zero_branch_weighted_mean_dict[each_column] = round((len(left_df)/len(df))* left_df['gradient'].mean(),4)

            elif len(list(set(df[each_column]))) == 1 and list(set(df[each_column]))[0] == 1:
                zero_branch_mean_dict[each_column] = 0
                zero_branch_weighted_mean_dict[each_column] = 0
        #print(zero_branch_weighted_mean_dict)
        #print(zero_branch_mean_dict)

    return zero_branch_mean_dict, zero_branch_weighted_mean_dict



def one_branch_variance_calculation(df):

    one_branch_mean_dict = {}
    one_branch_weighted_mean_dict = {}
    for each_column in df.columns:
        if each_column != 'regression_value' and each_column != 'gradient' and each_column != 'sigmoid_reg_value' and each_column != 'class':
            if len(list(set(df[each_column]))) == 2 or (len(list(set(df[each_column]))) == 1 and list(set(df[each_column]))[0] == 1):

                right_df = df[[each_column, "gradient"]][df[each_column] == 1]
                #print(each_column)
                #print("mean", right_df['gradient'].mean())
                #print("weighted_mean", (len(right_df) / len(df)) * right_df['gradient'].mean())
                one_branch_mean_dict[each_column] = round(right_df['gradient'].mean(),4)
                one_branch_weighted_mean_dict[each_column] = round((len(right_df)/len(df)) * right_df['gradient'].mean(),4)
            elif len(list(set(df[each_column]))) == 1 and list(set(df[each_column]))[0] == 0:
                one_branch_mean_dict[each_column] = 0
                one_branch_weighted_mean_dict[each_column] = 0

    return one_branch_mean_dict, one_branch_weighted_mean_dict




#print(zero_branch_variance_calculation(train_df)[1])
#print(zero_branch_variance_calculation(train_df)[0])


#print(one_branch_variance_calculation(train_df)[1])



def combine_mean(df):

    dict0_normal = zero_branch_variance_calculation(train_df)[0]
    dict0 = zero_branch_variance_calculation(train_df)[1]

    dict1_normal = one_branch_variance_calculation(train_df)[0]
    dict1 = one_branch_variance_calculation(train_df)[1]
    combine_mean_counter = {v: dict0.get(v, 0) + dict1.get(v, 0) for v in set(dict0)}
    #print(dict0_normal)
    #print(dict0)
    #print('\t')
    #print(dict1_normal)
    #print(dict1)
    #print('\t')
    #print(combine_mean_counter)

    return min(combine_mean_counter,key=combine_mean_counter.get),dict0_normal,dict1_normal

def predict(df,main_df,i):

    result_list = []
    inputs = combine_mean(df)
    best_split = inputs[0]
    for index,rows in test_df.iterrows():
        if test_df.loc[index,best_split] == 0:
            result_list.append(inputs[1][best_split])
        elif test_df.loc[index,best_split] == 1:
            result_list.append(inputs[2][best_split])

    main_df[i] = result_list
    #print(main_df)








def update(df):

    inputs = combine_mean(df)
    best_split = inputs[0]
    #print ("the best split is",best_split)
    for index,rows in df.iterrows():
        if df.loc[index,best_split] == 0:
            df.loc[index,'regression_value'] = df.loc[index,'regression_value'] + inputs[1][best_split]
        elif df.loc[index,best_split] == 1:
            df.loc[index, 'regression_value'] = df.loc[index, 'regression_value'] + inputs[2][best_split]
        #print ("the doubting parameter",df.loc[index,'sigmoid_reg_value'] )
        #print ("the sigmoid of doubting parameter",sigmoid(df.loc[index,'regression_value']))
        #break
        df.loc[index,'sigmoid_reg_value'] = sigmoid(df.loc[index,'regression_value'])
        df.loc[index, 'gradient'] = df.loc[index, 'class'] - df.loc[index, 'sigmoid_reg_value']
    #print(df)
    return df

def final_processor(df):


    df['final_regression_sum'] = df.sum(axis=1)
    for index,rows in df.iterrows():
        df.loc[index,'final_regression_sum'] = sigmoid(df.loc[index,'final_regression_sum'])
    df ['predicted_class_label'] = 0
    #print (df)
    df["predicted_class_label"][df['final_regression_sum'] > 0.4] = 1
    #print (df['predicted_class_label'].values.tolist())
    #print (df)
    return df['predicted_class_label'].values.tolist()

def confusion_matrix(df,number_of_iterations,accuracy_dict):
    """
    Function is used to calculate the confusion matrix. Takes 2 lists as input, the actual list with the list of actual class label
    and the predicted list with the list of predicted class labels.
    """

    actual_list = test_df['class'].values.tolist()
    predicted_list = final_processor(df)

    a = 0;b=0;c=0;d=0
    for i in range(0, len(actual_list)):
        if actual_list[i] == 0 and predicted_list[i] == 0:
            a += 1
        if actual_list[i] == 0 and predicted_list[i] == 1:
            b += 1
        if actual_list[i] == 1 and predicted_list[i] == 0:
            c += 1
        if actual_list[i] == 1 and predicted_list[i] == 1:
            d += 1
    print (pd.DataFrame([[a, b], [c, d]], columns=['No', 'Yes'],index=['No', 'Yes']))
    print ("The accuracy of the model is:",((a+d)/(a+b+c+d))*100)
    accuracy_dict[number_of_iterations] = ((a+d)/(a+b+c+d))*100
    return pd.DataFrame([[a, b], [c, d]], columns=['No', 'Yes'],index=['No', 'Yes'])


def main(df):

    accuracy_dict = {}
    for number_of_iterations in [5,10,15,20,25,30]:
        main_df = pd.DataFrame()
        for i in range(0,number_of_iterations):
            if i == 0:
                df['regression_value'] = 0
                df['sigmoid_reg_value'] = sigmoid(list(set (df['regression_value']))[0])
                df['gradient'] = df['class'] - df['sigmoid_reg_value']
            #print(df)
            predict(train_df,main_df, i + 1)
            update(df)
        confusion_matrix(main_df,number_of_iterations,accuracy_dict)
    print ('\t')
    print(accuracy_dict)


if __name__ == "__main__":
    train_df = pd.read_csv("/Users/vigneshsureshbabu/Documents/AML/PA2/mushrooms/agaricuslepiotatrain1.csv")
    train_df = train_df.rename(columns={'poisonous': 'class'})
    test_df = pd.read_csv("/Users/vigneshsureshbabu/Documents/AML/PA2/mushrooms/agaricuslepiotatest1.csv")
    test_df = test_df.rename(columns={'poisonous': 'class'})
    main(train_df)