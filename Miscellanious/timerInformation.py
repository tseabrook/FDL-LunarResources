from timeit import default_timer as timer
import numpy as np

def timerInformation(jobname, percentage, start, elapsed):
    #timerInformation updates and prints [elapsed] times for [jobname] with an estimated time remaining
    end = timer() #Timestamp
    elapsed += end - start #Time since last start call to timer() is added to overall elapsed time
    start = timer() #New timer() is started
    secs_remaining = ((elapsed * 100) / percentage) - elapsed #calculating overall job time minus elapsed
    hours_remaining = np.floor_divide(secs_remaining, 3600)
    secs_remaining = np.mod(secs_remaining, 3600)
    mins_remaining = np.floor_divide(secs_remaining, 60)
    secs_remaining = np.floor_divide(np.mod(secs_remaining, 60), 1)
    s = jobname+' is '+repr(percentage)+'% Complete. Estimated Time Remaining: ' #Print job and percentage completed
    if hours_remaining > 0: #Conditionally print hours remaining
        s += repr(np.int_(hours_remaining)) + 'h '
    if mins_remaining > 0: #Conditionally print minutes remaining
        s += repr(np.int_(mins_remaining)) + 'm '
    s += repr(np.int(secs_remaining)) + 's.' #Print time remaining
    s += ' Time Elapsed: ' + repr(np.int_(np.floor_divide(elapsed, 60))) + 'mins.' #Print time elapsed
    print(s)
    return start, elapsed #Return newly initialised timer and current elapsed time

