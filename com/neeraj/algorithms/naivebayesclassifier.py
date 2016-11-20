# -*- coding: utf-8 -*-
"""
Naive Bayes Classifier.
@author: neeraj
"""
import csv
import random
import math

class NaiveBayesClassifier:
    """ Class NaiveBayesClassifier. Implementation of Naive Bayes Classifier
    for numeric data set. Last attribute should be the class values. 
    NaiveBayesClassifier have the following properties:
    
    Attributes:
        train_set: Training data set.
        test_set: Testing data set.
        predictions: Predicted class value.
    
    Methods:
        test_model(): Tests accuracy of the model.
        load_data(): Load data from specified file and returns data set.
        split_data: Splits the data set to training set and testing set.
        data_by_class(): Grouping data records by their class value.
        class_summary(): Prepares summary of each class.
        summarize(): Summarize each column data.
        std_dev(): Calculates standard deviation of each column.
        mean(): Calculates mean of each column.
        calculate_probability(): Calculates probability of each value.
        calculate_class_probabilities(): Calculate probability of each class 
                                            for each record.
        perform_predictions(): Performs prediction on each test record.
        predict(): Predicts the class value.
        calculate_accuracy(): Calculates accuracy of prediction.
        print_predictions(): Prints the actual class value and predicted 
                                            class value.
        main(): Performs the predictions on given input files.
    
    """
    
    train_set = [[]]
    test_set = [[]]
    predictions = []
    
    def test_model(self, file_name, split_ratio):
        """Takes arguments 'file_name' and 'split_ratio'. Calculates the accuracy
        of prediction for a given data.
        """
        # Loads data set.
        data_set = self.load_data(file_name)
        # Splits data set to train set and test set.
        self.train_set, self.test_set = self.split_data(data_set, split_ratio)
        # Summarizing data set.
        summaries = self.class_summary()
        # Performing prediction.
        self.perform_predictions(summaries)
        # Prints prediction.
        self.print_predictions()
        # Calculates accuracy.
        self.calculate_accuracy()
    
    def load_data(self, file_name):
        """Takes argument 'file_name'. Loads data set.
        """
        data_set = list(csv.reader(open(file_name, "rb")))
        # Changing each value to float.
        for i in range(len(data_set)):
            data_set[i] = [float(value) for value in data_set[i]]
        return data_set
    
    def split_data(self, data_set, split_ratio):
        """Takes arguments 'data_set', 'split_ratio'. Splits data set to train set 
        and test set.
        """        
        train_size = int(len(data_set) * split_ratio)
        train = []
        test = list(data_set)
        # Splitting data in the ratio.
        while len(train) < train_size:
            index = random.randrange(len(test))
            train.append(test.pop(index))
        return [train, test]
    
    def data_by_class(self):
        """Takes no argument. Group/Map data by their class value.
        """
        class_data = {}
        for i in range(len(self.train_set)):
            record = self.train_set[i]
            if (record[-1] not in class_data):
                class_data[record[-1]] = []
            class_data[record[-1]].append(record)
        return class_data
    
    def class_summary(self):
        """Takes no argument. Prepares class summary. summary consists of mean and 
        standard deviation of each column/attribute.
        """
        # Getting data by class.
        class_data = self.data_by_class()
        summaries = {}
        # Preparing summary for each class.
        for class_value, class_records in class_data.iteritems():
            summaries[class_value] = self.summarize(class_records)
        return summaries
    
    def summarize(self, data_set):
        """Takes argument 'data_set' which is belongs to a particular class. 
        """
        # Summary of each column/attribute.
        summary = [(self.mean(column), self.std_dev(column)) for column in 
                   zip(*data_set)]
        # Deletes class attribute/column.
        del summary[-1]
        return summary
            
    def std_dev(self, column):
        """Takes argument 'column', each column of data set.
        """
        # Calculates mean of the column values.
        mean = self.mean(column)
        # Calculates variance of the column values.
        variance = sum([pow(x-mean,2) for x in column])/float(len(column)-1)
        # Returns standard deviation of the column values.
        return math.sqrt(variance)
    
    def mean(self, column):
        """Takes argument 'column', each column of data set. and returns mean.
        """
        return sum(column)/float(len(column))
    
    def calculate_probability(self, value, mean, std_dev):
        """Takes arguments 'value', 'mean', 'std_dev'. Each value in the data set 
        and the corresponding column mean and standard deviation
        """
        # Calculates the exponent.
        exponent = math.exp(-(math.pow(value-mean,2)/(2*math.pow(std_dev,2))))
        # Returns the probability.
        return (1 / (math.sqrt(2*math.pi) * std_dev)) * exponent
 
    def calculate_class_probabilities(self, summaries, test_record):
        """Takes arguments 'summaries', 'test_record'. Class summary and single 
        record from test set.
        """
        probabilities = {}
        for class_value, classSummaries in summaries.iteritems():
            probabilities[class_value] = 1
            for i in range(len(classSummaries)):
                # mean and std_dev of ith column/attribute.
                mean, std_dev = classSummaries[i]
                # ith column value of this record.
                value = test_record[i]
                # Calculating class probability by multiplying probability of 
                # each values.
                probabilities[class_value] *= self.calculate_probability(value, 
                                                                         mean, 
                                                                         std_dev)
        return probabilities
    
    def perform_predictions(self, summary):
        """Takes argument 'summary'. Performs prediction.
        """
        self.predictions = []
        for i in range(len(self.test_set)):
            # Predicted class value of each record in test set.
            prediction = self.predict(summary, self.test_set[i])
            self.predictions.append(prediction)

    def predict(self, summary, test_record):
        """Takes argument 'summary', 'test_record'. Predicted class is the class 
        value with maximum probability.
        """
        # Calculating class probabilities of 'test_record'.
        probabilities = self.calculate_class_probabilities(summary, test_record)
        predicted_class = -1
        max_probability = -1
        for class_value, probability in probabilities.iteritems():
            if predicted_class == -1 or probability > max_probability:
                max_probability = probability
                predicted_class = class_value
        return predicted_class
 
    def calculate_accuracy(self):
        """Takes no argument. Calculating accuracy by comparing the predicted class 
        value with actual class value.
        """
        true_predictions = 0
        for i in range(len(self.test_set)):
            # Checking for correct prediction.
            if self.test_set[i][-1] == self.predictions[i]:
                true_predictions+= 1
        print("\nAccuracy: {0}%").format((true_predictions/float(len(self.test_set)))
                                          * 100.0)
    
    def print_predictions(self):
        """Takes no argument. Printing Prediction.
        """
        print("No.\tClass\tPrediction")
        for i in range(len(self.test_set)):
            print("{0}\t{1}\t{2}").format(i,self.test_set[i][-1], self.predictions[i])
 
    def main(self, train_file, input_file):
        """Takes arguments 'train_file', the data file with which the model is to be 
        prepared and 'input_file', the data file for prediction.
        """
        # Loading training data.
        self.train_set = self.load_data(train_file)
        # Loading the input data.
        self.test_set = self.load_data(input_file)
        # Performs prediction.
        self.perform_predictions(self.class_summary())
        # Prints Prediction.
        self.print_predictions()

        
# Data set file.        
file_name = '../../neeraj/resource/pima-indians-diabetes.data.csv'
# Ratio in which data set is to be splitted for testing.
split_ratio = 0.70
# Testing model accuracy.
naiveBayes1 = NaiveBayesClassifier()
naiveBayes1.test_model(file_name, split_ratio)

print("\n---------new prediction---------- \n")
# classifier object.
naiveBayes2 = NaiveBayesClassifier()
train_file = '../../neeraj/resource/pima-indians-diabetes.data.csv'
# In case of the input file for prediction class attribute should not be empty, set the 
# class value to some numeric value. example: -1.
input_file = '../../neeraj/resource/input.csv' 
naiveBayes2.main(train_file, input_file)

    