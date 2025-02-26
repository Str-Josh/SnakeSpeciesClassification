""" AI Final Project -- Snake Species Classification
Created on Mon Apr  1 10:35:57 2024
@author: Joshua Carter
"""
import os
import math
import time
import typing

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from openpyxl import Workbook


class Data(object):
    def __init__(self, datafile_path: str, 
                 attributes_for_classification: list,
                 column_for_class_names: str):
        """

        Parameters
        ----------
        datafile_path : str
            file path for the dataset containing training, testing, 
            and category data for the classification.
            
        attributes_for_classification : list
            a list of attributes of snake to determine it's venomous status.
            
        column_for_class_names : str
            The name of the column to use for the classes within the dataset.

        Returns
        -------
        None.

        """
        # Path of the datafile (i.e., snakeData.xlsx).
        self.path = datafile_path
        
        # The Classification tab in the datafile.
        self.classifications = pd.read_excel(dataset_path, sheet_name="Classification")

        self.training_data = pd.read_excel(dataset_path, sheet_name="TrainingData")
        
        self.attributes = attributes_for_classification
        
        self.class_name_column = column_for_class_names
        
        return None
    
    
    def create_bayes_table(self):
        probabilities_tables_grouped_attribute = []
        bayes_table_rows = ["P(m)"]
        features = []
        classes  = []
        
        structured_data   = {}
        
        snake_attributes  = self.attributes
        training_data     = self.training_data
        NAME_COLUMN       = self.class_name_column
        alpha = 1
        
        for column_name in snake_attributes:
            # get array of unique values in pd.Series of each snake attribute
            unique_responses = np.unique(np.array(training_data[column_name].tolist()))
            structured_data[column_name] = unique_responses.tolist()
            
            # iterate thru unique values to create list of all types
            for unique in unique_responses:
                features.append(f"{column_name}_{unique}")
                bayes_table_rows.append(f"P({column_name}_{unique}|m)")


        # receive the classes in the dataframe
        classes = np.unique(np.array(training_data[NAME_COLUMN].tolist()))        

        
        # get values for priors / P(m)!
        class_probabilities = training_data.groupby(NAME_COLUMN).count()["HeadShape"] / len(training_data)
        for j in class_probabilities:
            probabilities_tables_grouped_attribute.append(j)


        # create empty 2d array to store the probabilities for our bayes table
        empty_data = np.zeros([ len(bayes_table_rows), len(classes) ])
        empty_data[0] = probabilities_tables_grouped_attribute
        
        result_dict = {}
        for snake_class_name, group_data in training_data.groupby(NAME_COLUMN):
            class_data = {}
            #print(group_data)
            
            for attribute in snake_attributes:
                class_data[attribute] = group_data[attribute].tolist()
                #y = group_data[attribute]
                value_counts = Counter(group_data[attribute])

                # get our conditional probabilities
                probs = [(value_counts[response_category] + alpha) / (
                    (alpha*len(structured_data[attribute])) + 
                         len(group_data[attribute])) 
                         for response_category in structured_data[attribute]]

                for _counter, j in enumerate(structured_data[attribute]):
                    col_index = classes.tolist().index(snake_class_name)
                    row_index = bayes_table_rows.index(f"P({attribute}_{j}|m)")
                    empty_data[row_index][col_index] = probs[_counter]
            
            result_dict[snake_class_name] = class_data
        
        self.bayes_table = pd.DataFrame( empty_data, 
                                        index = bayes_table_rows,
                                        columns = classes)

        #print(self.bayes_table)
        i_want_to = False
        if i_want_to:
            self.bayes_table.to_excel("BayesTable1.xlsx", sheet_name='Bayes Table', index=True)
        return self.bayes_table
    

class Classifier(object):
    def __init__(self, X: list,
                 naive_bayes_conditional_probabilities_table:pd.DataFrame,
                 classifications_data_file:pd.DataFrame):
        
        # x is our list of attributes from our given input
        self.X = X
        self.table = naive_bayes_conditional_probabilities_table
        self.classifications_data_file = classifications_data_file
        self.setup()
        return None
    
    
    def setup(self):
        table = self.table
        X = self.X
        probabilities = {}
        
        # calculate probabilities P(X|category1), P(X|category2), ...
        prob = 1
        for class_category in table:
            for given_feature in X:
                prob *= table[class_category].loc[f"P({given_feature}|m)"]
            probabilities[class_category] = prob
            prob = 1
        
        posteriors  = {}
        denominator = 0
        posterior_sum = 0
        x={}
        
        # calculate posteriors
        for class_name, conditional in probabilities.items():
            class_probability = table[class_name].loc["P(m)"]
            conditional = probabilities[class_name]
            term = conditional * class_probability
            denominator += term
        
        for class_name in table:
            x[class_name]=probabilities[class_name]*table[class_name].loc["P(m)"]/denominator
            posterior_sum += x[class_name]
        
        if posterior_sum < 0.999 or posterior_sum > 1.111:
            print("There was an error in the Posterior probability calculations")
        else:
            posteriors = pd.Series(data= x, index= [c for c in table])
            argmax_n_class = posteriors.idxmax()
            argmax_n = posteriors.loc[argmax_n_class]
            
            
            classif_df = self.classifications_data_file
            classif_filtered_df = classif_df[classif_df["species"] == argmax_n_class]
            
            venomous_classif = classif_filtered_df["venomous"].values[0]
            
            ven_cat = "VENOMOUS!" if venomous_classif==1 else "NOT venomous"
            
            print(ven_cat)
            print(f"The given input features describe a {argmax_n_class} " +
                  f"({ven_cat}) snake species with " +
                  f"probability of {round(argmax_n, 8)}")
            """
            print(f"The given input features describe a {ven_cat} " +
                  f"{argmax_n_class} species of snake with " +
                  f"probability of {round(argmax_n, 8)}")
            """
            print(posteriors)
            
        return (argmax_n_class, argmax_n), posteriors


def algo_run_timer(function):
    ## Used to determine runtimes of specfic methods for optimization.
    algo_timer = time.time()
    function_return_value = function()
    print(f"Function runtime: {round(time.time() - algo_timer, 6)}")
    return function_return_value


if __name__ == "__main__":
    program_timer = time.time()
    current_directory = os.getcwd()
    dataset_path = current_directory + "\\snakeData.xlsx"
    
    # Column names from the dataset
    attributes = ["BodyShape", "HeadShape", "Color", 
                  "BackPattern", "ScaleTexture", "EyePupilShape"]
    class_names_column = "Name"
    
    data = Data(dataset_path, attributes, class_names_column)
    naive_bayes_table = data.create_bayes_table()
    
    example_input0 = ["BodyShape_slender",
                     "ScaleTexture_smooth",
                     "Color_tan",
                     "BackPattern_striped"]  # hope to classify as Coral Snake
    
    example_input1 = ["BodyShape_stout",
                      "BackPattern_banded", 
                      "Color_tan",
                      "HeadShape_unknown"]  # hope to classify as Copperhead
    
    example_input2 = ["BodyShape_slender", 
                      "HeadShape_triangular",
                      "ScaleTexture_keeled"]
    
    example_input3 = ["BodyShape_typical",
                      "HeadShape_pointed",
                      "Color_tan"]

    classifier = Classifier(example_input1, naive_bayes_table, data.classifications)
    print(f"Time to completion: {time.time() - program_timer}")
