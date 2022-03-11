import numpy as np

def timer(start_time=None):
    from datetime import datetime
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def AMS(prediction, probability):
        cut_off = 0.85 
        prob_pred = probability[:,0]*prediction
        s = np.sum(probability[probability[:,1]>cut_off])
        b = np.sum(prob_pred[probability[:,1]<cut_off])

        b_reg = 10
        ams = np.sqrt(2 * ((s + b + b_reg) * np.log(1 + s / (b + b_reg)) - s))
        return ams