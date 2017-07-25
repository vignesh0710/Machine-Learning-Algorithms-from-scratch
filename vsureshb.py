from __future__ import division
import pandas as pd
import numpy as np
import math as m
import sys, os;

# boosting_test_test

def data_processing(df):

    """
    Function is used to pre process the training and test data
    """

    if entype == "bag":
        df = df.rename(columns={'bruises?-bruises': 'class'})
        del df['bruises?-no']
        return df
    else:
        df = df.rename(columns={'bruises?-bruises': 'class'})
        del df['bruises?-no']
        for index, rows in df.iterrows():
            if rows['class'] == 0:
                rows['class'] = -1
        return df


def function_fetch_actual(df):
    """
    Function is used to fetch the actual class labels of the data points of the test data into a list. Takes a dataframe as an input.
    """
    actual_list = []
    actual_list.append(df['class'].values.tolist())
    return actual_list


def get_column_list(df):
    """
    Function to return the list of columns from the data frame
    """
    header_list = []
    for each in df.columns:
        if each != 'id':
            header_list.append(each)

    return header_list

def class_based_gain(df):

    dist_df = df.groupby("class").count().values.tolist()
    process = []
    for each in dist_df:
       process.append(list(set(each)))
    for i in range(len(process)):
        process[i] = process[i][0]

    class_gain = 0.0
    if len(process) == 2:
        outer_dist_class0 = process[0] / sum(process)
        outer_dist_class1 = process[0] / sum(process)
        class_gain = -outer_dist_class0*np.log2(outer_dist_class0)  - outer_dist_class1*np.log2(outer_dist_class1)
        if class_gain == -0.0:
            class_gain = 0.0
    elif len(process) ==1:
        outer_dist_class = process[0] / sum(process)
        class_gain = -outer_dist_class * np.log2(outer_dist_class)
        if class_gain == -0.0:
            class_gain = 0.0

    return class_gain

def entropy_calculation(dist_list, df):
    """
    Function to calculate the entropy of the data. It takes a dataframe as the input
    """
    if entype == "bag":
        entropy = None
        for i in range(len(dist_list)):
            dist_list[i] = dist_list[i][0]
        if len(dist_list) == 2 and len(dist_list) != 0:
            entropy = -((dist_list[0] / sum(dist_list)) * np.log2(dist_list[0] / sum(dist_list)) + (
                (dist_list[1] / sum(dist_list)) * np.log2(dist_list[1] / sum(dist_list))))
        elif len(dist_list) == 1 and len(dist_list) != 0:
            entropy = -((dist_list[0] / sum(dist_list)) * np.log2(dist_list[0] / sum(dist_list)))
        elif len(dist_list) == 0:
            entropy = 0
        return entropy

    else:
        entropy = None
        if len(dist_list) == 2 and len(dist_list) != 0:
            if dist_list == [0.0, 0.0] or dist_list == [1.0, 0.0]:
                entropy = 0
            else:
                entropy = -1 * (((dist_list[0] / sum(dist_list)) * np.log2(dist_list[0] / sum(dist_list)) + (
                    (dist_list[1] / sum(dist_list)) * np.log2(dist_list[1] / sum(dist_list)))))
                entropy = (sum(dist_list) / len(df)) * entropy
        elif len(dist_list) == 1 and len(dist_list) != 0:
            entropy = -1 * ((dist_list[0] / sum(dist_list)) * np.log2(dist_list[0] / sum(dist_list)))
            entropy = (sum(dist_list) / len(df)) * entropy

        return entropy


def column_based_infogain_left(df):
    """
    Function to calculate the entropy for the nodes in the left branch of the tree.It takes a dataframe as the input
    """

    if entype == "bag":

        individual_entropy_list_left = []
        for each_column in df.columns:
            if each_column != 'class' and each_column != 'bruises?-no':
                df_left = df[[each_column, "class"]][df[each_column] == 0].groupby("class").count()
                df_left_sum = df_left.sum()
                entropy = 0.0
                outer_dist = df_left_sum[each_column] / len(df)
                for each_index in df_left.index:
                    inner_element = (df_left.loc[each_index, each_column] / df_left_sum[each_column])
                    entropy += inner_element * np.log2(inner_element)
                final_entropy = -entropy * outer_dist
                individual_entropy_list_left.append(final_entropy)
        return individual_entropy_list_left

    else:

        left_df = df
        header_list = get_column_list(left_df)
        individual_entropy_list_left = []
        for each_column in header_list:
            if each_column != 'class' and each_column != 'weight':
                left_child_entropy = []
                left_temp_df = pd.DataFrame(left_df[[each_column, "class", 'weight']][left_df[each_column] == 0],
                                            columns=[each_column, 'class', 'weight'])
                left_child_entropy.append(round(left_temp_df.loc[left_temp_df['class'] == 1, 'weight'].sum(), 2))
                left_child_entropy.append(round(left_temp_df.loc[left_temp_df['class'] == -1, 'weight'].sum(), 2))
                individual_entropy_list_left.append(entropy_calculation(left_child_entropy, left_df))
        return individual_entropy_list_left

def column_based_infogain_right(df):
    """
    Function to calculate the entropy for the nodes in the right branch of the tree.It takes a dataframe as the input
    """

    if entype == 'bag':
        individual_entropy_list_right = []
        for each_column in df.columns:
            if each_column != 'class' and each_column != 'bruises?-no':
                df_right = df[[each_column, "class"]][df[each_column] == 1].groupby("class").count()
                df_right_sum = df_right.sum()
                entropy = 0.0
                outer_dist = df_right_sum[each_column] / len(df)
                for each_index in df_right.index:
                    inner_element = (df_right.loc[each_index, each_column] / df_right_sum[each_column])
                    entropy += inner_element * np.log2(inner_element)
                final_entropy = -entropy * outer_dist
                individual_entropy_list_right.append(final_entropy)
        return individual_entropy_list_right

    else:
        right_df = df
        header_list = get_column_list(right_df)
        individual_entropy_list_right = []
        for each_column in header_list:
            if each_column != 'class' and each_column != 'weight':
                right_child_entropy = []
                right_temp_df = pd.DataFrame(right_df[[each_column, "class", 'weight']][right_df[each_column] == 1],
                                             columns=[each_column, 'class', 'weight'])
                right_child_entropy.append(round(right_temp_df.loc[right_temp_df['class'] == 1, 'weight'].sum(), 2))
                right_child_entropy.append(round(right_temp_df.loc[right_temp_df['class'] == -1, 'weight'].sum(), 2))
                individual_entropy_list_right.append(entropy_calculation(right_child_entropy, right_df))
        return individual_entropy_list_right


def get_best_column(df):
    """
    Function to add the  information gain for the nodes in both left and right branches of the tree and finds the best attribute.It takes a dataframe as the input
    """


    left_infogain = column_based_infogain_left(df)
    right_infogain = column_based_infogain_right(df)
    total_infogain = [x + y for x, y in zip(left_infogain, right_infogain)]
    header_list = get_column_list(df)
    header_list.remove("class")
    infogain_dict = dict(zip(header_list, total_infogain))

    return min(infogain_dict, key=infogain_dict.get)


class BinaryTreeNode(object):
    """
    class Declration for building the treee.
    node_attribute: value of the node
    left:value of the left child of the node
    right:value of the right child of the node
    parent: parent of the left and right child, (ie) the parent of the particular node
    class_label: class_label of the node, if the node is a leaf node
    """

    def __init__(self):
        self.node_attribute = None
        self.left = None
        self.right = None
        self.parent = None
        self.class_label = None


def splitting_the_dataframe(df):
    """
    This function is used to split the dataframe (in this case the nodes of the decision tree)
    into 2 nodes, the left node satisying the given condition and the right node not satisfying the given condition
        #dataframe : the dataframe which is to be split
    It is also used to find the best attribute in the left and right data frame, (ie) left and right child of a particular node.
    """
    header_list = get_column_list(df)
    root = get_best_column(df)
    left_data_frame = df[header_list][df[root] == 0]
    del left_data_frame[root]
    left_obj = BinaryTreeNode()
    left_obj.node_attribute = get_best_column(left_data_frame)
    left_obj.parent = root
    right_data_frame = df[header_list][df[root] == 1]
    del right_data_frame[root]
    right_obj = BinaryTreeNode()
    right_obj.node_attribute = get_best_column(right_data_frame)
    right_obj.parent = root

    return root, left_data_frame, left_obj, right_data_frame, right_obj


def build_tree(df, class_obj, depth):
    """
    The function is used to build the tree recursively. Takes the dataframe, class object and the depth of the tree to be built

    """
    if depth == 0:
        class_obj.class_label = majority_class_label(df)
        return class_obj

    else:
        class_obj.node_attribute = splitting_the_dataframe(df)[0]
        left_data_frame = splitting_the_dataframe(df)[1]
        class_obj.left = splitting_the_dataframe(df)[2]
        right_data_frame = splitting_the_dataframe(df)[3]
        class_obj.right = splitting_the_dataframe(df)[4]
        if leaf_node_check(left_data_frame) == False:
            build_tree(left_data_frame, class_obj.left, depth - 1)
        elif leaf_node_check(left_data_frame) == True:
            class_obj.class_label = majority_class_label(left_data_frame)
        if leaf_node_check(right_data_frame) == False:
            build_tree(right_data_frame, class_obj.right, depth - 1)
        elif leaf_node_check(right_data_frame) == True:
            class_obj.class_label = majority_class_label(right_data_frame)

        return class_obj


def leaf_node_check(df):
    """
    Function to check whether a particular node is a leaf node.Takes dataframe as input.

    """
    class_count = set(df['class'].values.tolist())
    if len(class_count) == 2:
        return False
    else:
        return True


def majority_class_label(df):
    """
    Function to determine the class label of the node if a particular node is a leaf node or when the maximum depth is reached using the majority class approach. Takes dataframe as input.

    """
    if entype == "bag":
        count_1 = list(df['class']).count(1)
        count_0 = list(df['class']).count(0)
        if count_1 > count_0:
            majority_class = 1
        else:
            majority_class= 0

        return majority_class
    else:
        count_1 = list(df['class']).count(1)
        count_minus1 = list(df['class']).count(-1)
        if count_1 > count_minus1:
            majority_class = 1
        else:
            majority_class = -1
        return majority_class


def confusion_matrix(predicted_list, actual_list):
    """
    Function is used to calculate the confusion matrix. Takes 2 lists as input, the actual list with the list of actual class label
    and the predicted list with the list of predicted class labels.
    """
    if entype == "bag":
        a = 0;b=0;c=0;d=0
        for i in range(0, len(actual_list)):
            if actual_list[i] == 0 and predicted_list[i] == 1:
                a += 1
            if actual_list[i] == 0 and predicted_list[i] == 0:
                b += 1
            if actual_list[i] == 1 and predicted_list[i] == 1:
                c += 1
            if actual_list[i] == 1 and predicted_list[i] == 0:
                d += 1
    else:
        a = 0;b = 0;c = 0;d = 0
        for i in range(0, len(actual_list)):
            if actual_list[i] == -1 and predicted_list[i] == 1:
                a += 1
            if actual_list[i] == -1 and predicted_list[i] == -1:
                b += 1
            if actual_list[i] == 1 and predicted_list[i] == -1:
                c += 1
            if actual_list[i] == 1 and predicted_list[i] == 1:
                d += 1
    return pd.DataFrame([[a, b], [c, d]], columns=['No', 'Yes'],index=['No', 'Yes'])



def predict(df, tree, rows):
    """
    Function used to predict the class labels of the test data set using the tree built, using recursion
    """

    if tree.class_label != None:
        #print ("tree.class_label",tree.class_label)
        return tree.class_label
    else:
        if rows[tree.node_attribute] == 0:
            #print ("Visiting left", tree.node_attribute)
            class_label = predict(df, tree.left, rows)
            #print ("left_class_label",class_label)
        elif rows[tree.node_attribute] == 1:
            #print ("Visiting right", tree.node_attribute)
            class_label = predict(df, tree.right, rows)
            #print ("right_class_label", class_label)
        return class_label


def predicted_class_label(tree, main_df, i):
    """
    Calling the predicted function for each row in the test data set.Takes the object of the build_tree function, main_df(a dataframe for
    storing all the individual results of bagging and boosting, i is a counter for storing the individual results in the main_df
    """
    result_list = []
    for index, rows in test_df.iterrows():
        e = predict(test_df, tree, rows)
        result_list.append(e)
    main_df[i] = result_list
    #return confusion_matrix(result_list, actual_list[0])


def epsilon_calculator(df):
    """
    Function used to calculate the epsilon value by adding the weights of  misclassified data points during the various iterations of the adaboost algorithm.takes data frame as the input
    """
    epsilon = 0.0
    for index, rows in df.iterrows():
        if rows['class'] != rows['predicted_column']:
            epsilon = epsilon + rows['weight']
    normalization_constant = 2 * (np.sqrt((1 - epsilon) * epsilon))

    return round(epsilon, 2), normalization_constant


def alpha_calculator(epsilon):
    """
    Function used to calculate the alpha value using the epsilon value for boosting. Takes epsilon as the input
    """
    alpha = 0.5 * np.log((1 - epsilon) / epsilon)
    return alpha

def boosting_prediction(tree, df,main_df,i):
    """
    Function used to find the misclassified data points in an individual iteration of the boosting algorithm by predicting on the training data.
    Using that calculate epsilon, alpha calling the previous 2 functions. Find the weight, normalize it and update the
    weight for the next iteration for boosting. Core function for boosting.

    """
    alpha_list = []
    for index, rows in train_df.iterrows():
        boost_pred = predict(train_df, tree, rows)
        alpha_list.append(boost_pred)
    df['predicted_column'] = alpha_list
    epsilon = epsilon_calculator(df)[0]
    norm_constant = epsilon_calculator(df)[1]
    alpha = alpha_calculator(epsilon)
    # print ("alpha",alpha)
    correct_classified_weight = m.exp(-alpha)
    wrongly_classified_weight = m.exp(alpha)
    df["weight"][df["class"] != df["predicted_column"]] *= wrongly_classified_weight / norm_constant
    df["weight"][df["class"] == df["predicted_column"]] *= correct_classified_weight / norm_constant
    main_df[i] = main_df[i] * alpha
    return df


def create_bags(df, nummodels):
    """
    Function  used to create bags(samples of training data) for creating the ensembles for the bagging algorithm
    """

    bags_list = []
    for i in range(0, nummodels):
        bags = pd.DataFrame(df.values[np.random.randint(len(df), size=len(df))], columns=df.columns)
        bags_list.append(bags)
    return bags_list


def final_bagging_processor(main_df):

    final_list = []
    for index, rows in main_df.iterrows():
        if list(rows).count(1) > list(rows).count(0):
            final_list.append(1)
        else:
            final_list.append(0)
    actual_list = function_fetch_actual(test_df)
    return confusion_matrix(final_list,actual_list[0])


def final_boosting_processor(main_df):

    final_list = []
    for index,rows in main_df.iterrows():
        if sum(list(rows)) > 0:
            final_list.append(1)
        else:
            final_list.append(-1)
    #print final_list
    actual_list = function_fetch_actual(test_df)
    return confusion_matrix(final_list, actual_list[0])

def main(depth, nummodels, entype = "boost"):
    """
    main function used to call all the required functions above
    """

    if entype == "bag":
        main_df = pd.DataFrame()
        bags_list = create_bags(train_df, nummodels)
        for i in range(0, nummodels):
            class_obj = BinaryTreeNode()
            tree = build_tree(bags_list[i], class_obj, depth)
            predicted_class_label(tree, main_df, i + 1)
        print ("The confusion matrix for depth:", depth, "and number of bags", nummodels)
        print(final_bagging_processor(main_df))

    else:
        train_df['weight'] = 1 / len(train_df)
        #print (train_df)
        main_df = pd.DataFrame()
        for i in range(0, nummodels):
            class_obj = BinaryTreeNode()
            tree = build_tree(train_df, class_obj, depth)
            predicted_class_label(tree, main_df, i + 1)
            boosting_prediction(tree, train_df,main_df,i+1)
        print ("The confusion matrix for depth:", depth, "and number of trees", nummodels)
        print (final_boosting_processor(main_df))



def learn_bagged(tdepth, numbags, datapath):
    '''
    Function: learn_bagged(tdepth, numbags, datapath)
    tdepth: (Integer) depths to which to grow the decision trees
    numbags: (Integer)the number of bags to use to learn the trees
    datapath: (String) the location in memory where the data set is stored

    This function will manage coordinating the learning of the bag ensemble.

    Nothing is returned, but the accuracy of the learned ensemble model is printed
    to the screen.
    '''

    main(tdepth, numbags, entype = "bag")

def learn_boosted(tdepth, nummodels, datapath):
    '''
    Function: learn_boosted(tdepth, numtrees, datapath)
    tdepth: (Integer) depths to which to grow the decision trees
    numtrees: (Integer) the number of boosted trees to learn
    datapath: (String) the location in memory where the data set is stored

    This function wil manage coordinating the learning of the boosted ensemble.

    Nothing is returned, but the accuracy of the learned ensemble model is printed
    to the screen.
    '''

    main(tdepth, nummodels)
    # function call

if __name__ == "__main__":
    # The arguments to your file will be of the following form:
    # <ensemble_type> <tree_depth> <num_bags/trees> <data_set_path>
    # Ex. bag 3 10 mushrooms
    # Ex. boost 1 10 mushrooms

    # Get the ensemble type
    entype = sys.argv[1];
    # Get the depth of the trees
    tdepth = int(sys.argv[2]);
    # Get the number of bags or trees
    nummodels = int(sys.argv[3]);
    # Get the location of the data set
    datapath = sys.argv[4];
    train_file = "{}//agaricuslepiotatrain1.csv".format(datapath)
    test_file = "{}//agaricuslepiotatest1.csv".format(datapath)
    train_df = data_processing(pd.read_csv(train_file))
    test_df = data_processing(pd.read_csv(test_file))

    # Check which type of ensemble is to be learned
    if entype == "bag":
        # Learned the bag decision tree ensemble
        learn_bagged(tdepth, nummodels, datapath);
    else:
        # Learned the boosted decision tree ensemble
        learn_boosted(tdepth, nummodels, datapath);



