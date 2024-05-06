""" AI Final Project -- Snake Species Classification
Created on Mon Apr  1 10:35:57 2024
@author: Joshua Carter
"""
from PIL import Image
from IPython.display import display


import os  # accessing image from file manager
import math
import time  # performance
import typing  # readability

import pywt  # filtering input image
import torch  # classification

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from collections import Counter
from openpyxl import Workbook

from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder


class Camera(object):
    def __init__(self):
        # could be scaled to allow this to start a camera process and 
        # take an image for an application
        return None
    
    
    
    def upload_image(self, image_path: str):
        
        if os.path.exists(image_path):
            self.image = cv2.imread(image_path)
            
            # Converting image to grayscale if necessary
            """Not needed yet
            if len(self.image.shape) == 3:
                self.image_color = "grey"
                self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            """
        else:
            raise FileNotFoundError("The image file does not exist.")
        
        return None
    
    
    def display_image(self):
        cv2.imshow("Camera Image", self.image)
        return None
    
    
    def display_image_PIL(self):
        # because my Spyder won't disp img with cv2 so I'm using pillow
        # PILcolor_coverted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        correct_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(correct_img)
        pil_image.show()
        return None
    
    
    def show_image(self, image):
        correct_img = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(correct_img)
        pil_image.show()
        return None
    
    
    def preprocess_image(self):
        #self.new_image = math.sin(self.image + 70)
        #print(self.image[1][1:5])
        #print(self.new_image[1][1:5])
        #self.show_image(self.new_image)
        image = self.image
        """DWT and image enhancement stuff"""
        
        feature_matrix = np.zeros((image.shape[0], image.shape[1]))
        for i in range(0, image.shape[0]):
            for j in range(0,image.shape[1]):
                feature_matrix[i][j] = (
                    (int(image[i,j,0])+int(image[i,j,1])+int(image[i,j,2])) /3)
        print(feature_matrix)
        return None
    
    
class Data(object):
    def __init__(self, datafile_path: str, 
                 attributes_for_classification: list,
                 column_for_class_names: str):
        """

        Parameters
        ----------
        datafile_path : str
            file path for the dataset containing
            training, testing, categories data for the classification.

        Returns
        -------
        None.

        """
        self.path = datafile_path
        #self.excel_data = pd.read_excel(dataset_path, sheet_name=["Classification"])
        self.classifications = pd.read_excel(dataset_path, sheet_name="Classification")
        #self.training_data = pd.read_excel(dataset_path, sheet_name="TrainingData")
        #self.example_training_data = pd.read_excel(dataset_path, sheet_name="ExampleTrainingData(OLD)")
        self.training_data = pd.read_excel(
            dataset_path, sheet_name="TrainingData")
        self.attributes = attributes_for_classification
        self.class_name_column = column_for_class_names
        """
        self.attributes = ["TriangularHead", "SlitPupils", 
                           "ThickBodies", "PitBehindNose", 
                           "BodyBottomColor"]
        """
        """
        ["Image", "Name", "Venomous",
                               "BodyBottomPattern", "bodyTopPattern",
                               "ScaleTexture", "BodyBottomAnalPlate"]
        """
        return None
    
    
    def clean_data(self):
        # retrieve the data from the excel file
        # since we got shitty data, we wanna clean it
        return None
    
    
    def cross_validate(self, num_folds=10):
        
        """
        data = self.training_data  # Your data with features and target variable
        target = data["Name"]
        
        clf = CategoricalNB()
        cv = StratifiedKFold(n_splits = num_folds, 
                             shuffle = True, 
                             random_state = 42)
        
        label_encoder = LabelEncoder()
        label_encoder.fit(data["Name"])
        encoded_resp_var = label_encoder.transform(data["Name"])
        
        label_encoder.fit(data["BodyShape"])
        encoded_ind_var1 = label_encoder.transform(data["BodyShape"])
        label_encoder.fit(data["HeadShape"])
        encoded_ind_var2 = label_encoder.transform(data["HeadShape"])
        label_encoder.fit(data["Color"])
        encoded_ind_var3 = label_encoder.transform(data["Color"])
        label_encoder.fit(data["BackPattern"])
        encoded_ind_var4 = label_encoder.transform(data["BackPattern"])
        label_encoder.fit(data["ScaleTexture"])
        encoded_ind_var5 = label_encoder.transform(data["ScaleTexture"])
        label_encoder.fit(data["EyePupilShape"])
        encoded_ind_var6 = label_encoder.transform(data["EyePupilShape"])
        
        print(encoded_ind_var1, encoded_ind_var2,
              encoded_ind_var3, encoded_ind_var4,
              encoded_ind_var5, encoded_ind_var6, sep="\n")
        
        
        data_copy = data.copy()
        data_copy["Name"]          = encoded_resp_var
        data_copy["BodyShape"]     = encoded_ind_var1
        data_copy["HeadShape"]     = encoded_ind_var2
        data_copy["Color"]         = encoded_ind_var3
        data_copy["BackPattern"]   = encoded_ind_var4
        data_copy["ScaleTexture"]  = encoded_ind_var5
        data_copy["EyePupilShape"] = encoded_ind_var6
        
        # Perform cross-validation
        scores = []
        
        for train_index, test_index in cv.split(data_copy, target):
            X_train, X_test = data_copy.iloc[train_index], data_copy.iloc[test_index]
            y_train, y_test = target.iloc[train_index], target.iloc[test_index]
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)
            scores.append(score)
        
        # Print average accuracy across folds
        print(f"Average Accuracy: {sum(scores) / len(scores)}")
        """
        """
        x = data_copy[2:3]
        x["BodyShape"] = 2
        x["HeadShape"] = 3
        x["Color"] = 4
        x["BackPattern"] = 1
        x["ScaleTexture"] = 0
        x["EyePupilShape"] = 0
        print(clf.predict(x))
        """
        #print(clf.get_params())
        return None
    
    
    def create_bayes_table(self):
        probabilities_tables_grouped_attribute = []
        bayes_table_rows = ["P(m)"]
        features = []  # "column_name_unique" - ex. BodyShape_slender
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
                #print(probs)
                """
                probs = [(value_counts[response_category] + alpha) / (
                    (alpha) + 
                         len(group_data[attribute])) 
                         for response_category in structured_data[attribute]]
                #print(probs)
                """
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
        
        # calculate probabilities P(X|Physics), P(X|Biology), ...
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
            classif_filtered_df = classif_df[
                classif_df["species"] == argmax_n_class]
            
            
            poisonous_classif = classif_filtered_df["poisonous"].values[0]
            
            poi_cat = "POISONOUS!" if poisonous_classif==1 else "NON poisonous"
            
            print(poi_cat)
            print(f"The given input features describe a {poi_cat} " +
                  f"{argmax_n_class} species of snake with " +
                  f"probability of {round(argmax_n, 8)}")
            print(posteriors)
            
        return (argmax_n_class, argmax_n), posteriors


def algo_run_timer(function):
    algo_timer = time.time()
    function_return_value = function()
    print(f"Function runtime: {round(time.time() - algo_timer, 6)}")
    return function_return_value


if __name__ == "__main__":
    program_timer = time.time()
    
    current_directory = os.getcwd()
    
    # image_path = current_directory + "\\southernCopperhead.jpg"
    dataset_path = current_directory + "\\dataset\\snakeData.xlsx"

    # cam = Camera()
    # cam.upload_image(image_path)
    # cam.display_image_PIL()
    # cam.preprocess_image()
    
    #attributes = ["TriangularHead", "SlitPupils", "ThickBodies", "PitBehindNose", "BodyBottomColor"]
    attributes = ["BodyShape", "HeadShape", "Color", 
                  "BackPattern", "ScaleTexture", "EyePupilShape"]
    class_names_column = "Name"
    
    data = Data(dataset_path, attributes, class_names_column)
    naive_bayes_table = data.create_bayes_table()
    
    example_input0 = ["BodyShape_slender",
                     "ScaleTexture_smooth",
                     "Color_tan",
                     "BackPattern_striped"]  # hope to classify as Coral Snake
    example_input = ["ScaleTexture_keeled",
                     "EyePupilShape_unknown",
                     "Color_red_black_yellow"]
    a = ["BodyShape_stout", "HeadShape_triangular"]
    ex = ["BodyShape_slender", "HeadShape_triangular",
          "ScaleTexture_keeled"]
    #classifier = Classifier(ex, naive_bayes_table, data.classifications)
    """
    classifier = Classifier(["BodyShape_slender", "HeadShape_triangular",
                             "ScaleTexture_keeled"], 
                            naive_bayes_table, data.classifications)
    
    """
    classifier = Classifier(["BodyShape_typical","HeadShape_pointed",
                             "Color_tan"], 
                            naive_bayes_table, data.classifications)
    #data.cross_validate()
    
    cam = Camera()
    print(f"Time to completion: {time.time() - program_timer}")
