""" AI Final Project -- Snake Species Classification
Created on Mon Apr  1 10:35:57 2024
@author: Joshua Carter
"""
# do we need this shit
from PIL import Image
from IPython.display import display


import os  # accessing image from file manager
import math
import time  # performance
import typing  # readability

import pywt  # filtering input image
import torch  # classification

import cv2  # create input image into an object
import numpy as np  # do things with arrays :)
import pandas as pd  # idk
import matplotlib.pyplot as plt

from collections import Counter


class Camera(object):
    def __init__(self):
        # could be scaled to allow this to start a camera process and 
        # take an image for an application
        return None
    
    
    def upload_image(self, image_path: str):
        """

        Parameters
        ----------
        image_path : str
            File Path for a snake image.

        Raises
        ------
        FileNotFoundError
            The file path for the snake image could not be found.

        Returns
        -------
        None.

        """
        
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
        # because my IDE is dumb and won't disp img with cv2 so i use pil
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
        self.new_image = math.sin(self.image + 70)
        print(self.image[1][1:5])
        print(self.new_image[1][1:5])
        self.show_image(self.new_image)
        """DWT and image enhancement stuff"""
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
            dataset_path, sheet_name="ExampleTrainingData")
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
    
    
    def create_bayes_table(self):
        probabilities_tables_grouped_attribute = []
        bayes_table_rows = ["P(m)"]
        features = []  # "column_name_unique" - ex. BodyShape_slender
        classes  = []
        
        structured_data   = {}
        
        snake_attributes  = self.attributes
        training_data     = self.training_data
        NAME_COLUMN       = self.class_name_column
        
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

        
        # get values for P(m)!
        #z = training_data.groupby("Name").count()["TriangularHead"] / len(training_data) 
        class_probabilities = training_data.groupby(NAME_COLUMN).count()["HeadShape"] / len(training_data)
        for j in class_probabilities:
            probabilities_tables_grouped_attribute.append(j)


        # create empty 2d array to store the probabilities for our bayes table
        empty_data = np.zeros([ len(bayes_table_rows), len(classes) ])
        empty_data[0] = probabilities_tables_grouped_attribute
        #######################################################################
        
        result_dict = {}
        for snake_class_name, group_data in training_data.groupby(NAME_COLUMN):
            class_data = {}
            
            for attribute in snake_attributes:
                class_data[attribute] = group_data[attribute].tolist()
                #y = group_data[attribute]
                value_counts = Counter(group_data[attribute])
                probs = [
                    value_counts[response_category] /len(group_data[attribute])
                    for response_category in structured_data[attribute]]
                
                for _counter, j in enumerate(structured_data[attribute]):
                    col_index = classes.tolist().index(snake_class_name)
                    row_index = bayes_table_rows.index(f"P({attribute}_{j}|m)")
                    empty_data[row_index][col_index] = probs[_counter]
            
            result_dict[snake_class_name] = class_data
        
        self.bayes_table = pd.DataFrame( empty_data, 
                                        index = bayes_table_rows,
                                        columns = classes)
        
        print(self.bayes_table)
        return self.bayes_table
    

class Classifier(object):
    def __init__(self):
        return None


def algo_run_timer(function):
    algo_timer = time.time()
    function_return_value = function()
    print(f"Function runtime: {round(time.time() - algo_timer, 6)}")
    return function_return_value


if __name__ == "__main__":
    program_timer = time.time()
    current_directory = os.getcwd()
    image_path = current_directory + "\\southernCopperhead.jpg"
    dataset_path = current_directory + "\\dataset\\snakeData.xlsx"

    cam = Camera()
    cam.upload_image(image_path)
    # cam.display_image_PIL()
    # cam.preprocess_image()
    #attributes = ["TriangularHead", "SlitPupils", "ThickBodies", "PitBehindNose", "BodyBottomColor"]
    attributes = ["BodyShape", "HeadShape", "Color", "BackPattern", "ScaleTexture", "EyePupilShape"]
    class_names_column = "Name"
    data = Data(dataset_path, attributes, class_names_column)
    #BodyShape	HeadShape	Color	BackPattern	ScaleTexture	EyePupilShape
    algo_run_timer(data.create_bayes_table)
    print(f"Time to completion: {time.time() - program_timer}")
