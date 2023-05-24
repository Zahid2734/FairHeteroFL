import numpy as np
import matplotlib.pyplot as plt
import sys
import statistics as st
from utils.functions_new import *



def get_stats(data):
    # Calculate mean and standard deviation
    data = np.array(data)
    mean = np.mean(data)
    std = np.std(data)

    # Calculate variance
    var = np.var(data)

    # Format results as strings with desired format
    mean_str = f"{mean:.2f}±{std:.2f}"
    var_str = f"{var:.2f}±{np.sqrt(var):.2f}"

    # Return all values in a dictionary
    return {'mean': mean_str, 'variance': var_str}

def Full_report(data):
    print ('Group1 Stat',get_stats(data[3][-1]))
    print ('Group2 Stat',get_stats(data[5][-1]))
    print ('Group3 Stat',get_stats(data[7][-1]))
    print ('Group4 Stat',get_stats(data[9][-1]))
    print ('Group5 Stat',get_stats(data[11][-1]))


if __name__ == "__main__":
 # Access the values of q, q1, q2, q3, q4, q5
    file_name = sys.argv[1]
    Dataset0= open_file(file_name)
    Full_report(Dataset0)