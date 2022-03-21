import os
import numpy as np
import csv
import math
import pandas as pd


def timer(start_time=None):
    from datetime import datetime
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def create_solution_dictionary(solution):
    """ Read solution file, return a dictionary with key EventId and value (weight,label).
    Solution file headers: EventId, Label, Weight """
    
    solnDict = {}
    with open(solution, 'rb') as f:
        soln = csv.reader(f)
        soln.next() # header
        for row in soln:
            if row[0] not in solnDict:
                solnDict[row[0]] = (row[1], row[2])
    return solnDict

def write_to_csv(id, proba, threshold, name):
    labels = np.where(proba > threshold, "s", "b")
    rank_orders = np.argsort(proba) + 1 
    df_submission = pd.DataFrame({'EventId': id,    
                                    'RankOrder': rank_orders,
                                    'Class': labels})
    df_submission.to_csv(name, index=False)
        
