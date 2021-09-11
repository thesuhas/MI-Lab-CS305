'''
Assume df is a pandas dataframe object of the dataset given
'''

import numpy as np
import pandas as pd
import math
import random


'''Calculate the entropy of the entire dataset'''
# input:pandas_dataframe
# output:int/float
def get_entropy_of_dataset(df):

    # Array for value of classes
    count = [len(df[df[df.columns[-1]] == i]) for i in df[df.columns[-1]].unique()]
    total = len(df)

    # Convert array into fractions
    fracs = [(i / total) for i in count]

    # Get entropy
    entropy = 0
    for i in fracs:
        if i != 0:
            entropy += (- i * math.log2(i))

    return entropy


'''Return avg_info of the attribute provided as parameter'''
# input:pandas_dataframe,str   {i.e the column name ,ex: Temperature in the Play tennis dataset}
# output:int/float
def get_avg_info_of_attribute(df, attribute):
    
    # DF consisting only of that attribute
    new = df[[attribute, df.columns[-1]]]
   
    # Calculate entropy for each value that the attribute takes
    entropy = dict()

    # For each value that the attribute takes, calculate entropy
    for i in new[attribute].unique():
        entropy[i] = get_entropy_of_dataset(new[new[attribute] == i])
    
    avg_info = 0

    # For each value that the attribute takes
    for i in entropy:
        # Get df for each value that the attribute takes
        temp = new[new[attribute] == i]

        # Get attribute == value for each target class
        vals = [len(temp[temp[temp.columns[-1]] == u]) for u in new[new.columns[-1]].unique()]

        # Get the frac for each value that the attribute takes
        frac = sum(vals) / len(df)

        avg_info += (frac * entropy[i])

    return avg_info


'''Return Information Gain of the attribute provided as parameter'''
# input:pandas_dataframe,str
# output:int/float
def get_information_gain(df, attribute):

    return get_entropy_of_dataset(df) - get_avg_info_of_attribute(df, attribute)


#input: pandas_dataframe
#output: ({dict},'str')
def get_selected_attribute(df):
    '''
    Return a tuple with the first element as a dictionary which has IG of all columns 
    and the second element as a string with the name of the column selected

    example : ({'A':0.123,'B':0.768,'C':1.23} , 'C')
    '''
    max = -math.inf
    col = None

    ig_vals = dict()

    for i in df.columns[:-1]:
        # Get information gain for each attribute
        ig = get_information_gain(df, i)
        ig_vals[i] = ig
        # If it is higher than current max ig
        if ig > max:
            max = ig
            col = i
    
    return (ig_vals, col)