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

    # Unique elements of class
    unique = df[df.columns[-1]].unique()

    # Array for value of classes
    count = [len(df[df[df.columns[-1]] == i]) for i in unique]
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

    for i in new[attribute].unique():
        entropy[i] = get_entropy_of_dataset(new[new[attribute] == i])
    
    avg_info = 0

    for i in entropy:
        temp = new[new[attribute] == i]

        # Get value for every class
        vals = [len(temp[temp[temp.columns[-1]] == u]) for u in new[new.columns[-1]].unique()]

        # Get the frac
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
        ig = get_information_gain(df, i)
        ig_vals[i] =ig
        if ig > max:
            max = ig
            col = i
    
    return (ig_vals, col)
        

outlook = 'overcast,overcast,overcast,overcast,rainy,rainy,rainy,rainy,rainy,sunny,sunny,sunny,sunny,sunny'.split(
        ',')
temp = 'hot,cool,mild,hot,mild,cool,cool,mild,mild,hot,hot,mild,cool,mild'.split(
        ',')
humidity = 'high,normal,high,normal,high,normal,normal,normal,high,high,high,high,normal,normal'.split(
        ',')
windy = 'FALSE,TRUE,TRUE,FALSE,FALSE,FALSE,TRUE,FALSE,TRUE,FALSE,TRUE,FALSE,FALSE,TRUE'.split(
        ',')
play = 'yes,yes,yes,yes,yes,yes,no,yes,no,no,no,no,yes,yes'.split(',')
dataset = {'outlook': outlook, 'temp': temp,
               'humidity': humidity, 'windy': windy, 'play': play}
df = pd.DataFrame(dataset, columns=[
                      'outlook', 'temp', 'humidity', 'windy', 'play'])

print(get_selected_attribute(df))