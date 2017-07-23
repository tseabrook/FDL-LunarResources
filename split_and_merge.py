import numpy as np

#Split-and-Merge algorithm for turning edges into lines

#This algorithm is particularly useful for pattern recognition;
#It does hold requirements of continuity similarly to spline approximations.

#For the problem statement:
#Given a set of points S = [x_i y_i | i - 1,2,...,N]
#determine the minimum number n such that S is divided in n subsets S_1,S_2,...,S_n,
#Where on each of them the data points are approximated by a polynomial
#of order at most m-1 with an error norm less than a pre-specified quantity e.

#For this application, we will limit the solution to
#Piecewise linear approximation i.e. m = 2

class Line:
    def __init__(self, data, slope, intercept, mean, error):
        self.data = data
        self.slope = slope
        self.intercept = intercept
        self.mean = mean
        self.min = np.zeros(2)
        self.max = np.zeros(2)
        self.error = error

def point_line_distance(data, slope, intercept):
    num_dim = len(data)
    L1_distance = np.zeros((num_dim, len(data[0])), dtype=np.float64)
    # Find distance between data and line
    # 1) Take line perpendicular to slope (a = -slope)
    #   a2 = -(1/slope)
    # 2) Find y intercept for data on perpendicular line
    #   b2 = y - ax
    # 3) Find intersection between original and perpendicular lines
    # a1*x1 + b1 = a2*x2 + b2
    # (a1-a2)x = (b2-b1)
    # x = (b2-b1)/(a1-a2)
    # y = a1x + b1

    #   b2 = (data[1:num_dim] - (-slope * data[0]))
    #   b1 = intercept
    #   a1 = slope
    #   a2 = a

    a2 = -np.divide(1,slope)
    b2 = data[1:num_dim]-(np.multiply(a2,data[0]))
    # a1*x1 + b1 = a2*x2 + b2
    x2 = np.divide(intercept - b2, a2 - slope)

    #x2 = np.divide((data[1:num_dim] - (np.multiply(a2, data[0])) - intercept),(slope - a2))
    y2 = (slope * x2) + intercept
    #x2 = ((data[1:num_dim] + (slope * data[0])) - intercept) / (slope + (np.ones(num_dim-1) / slope))
    #y2 = slope * x2 + intercept

    L1_distance[1:num_dim,:] = data[1:num_dim] - y2
    L1_distance[0,:] = data[0] - x2
    L2_distance = np.sqrt(np.sum(np.mean(np.power(L1_distance,2),1)))
    return L2_distance

def line_of_best_fit(data):
    mean = np.mean(data, axis = 1)
    num_dim = len(mean)

    covariance = np.zeros((num_dim, num_dim), dtype=np.float64)
    for m in range(num_dim):
        for n in range(num_dim):
            if (m > n):
                covariance[m,n] = covariance[n,m]
            else:
                covariance[m,n] = np.mean(np.multiply((data[m] - mean[m]),(data[n] - mean[n])))
                if(covariance[m,n] == 0):
                    covariance[m,n] = 0.00000000001

    slope = covariance[1:num_dim,0] / covariance[0,0],
    intercept = mean[1:num_dim] - np.multiply(slope, mean[0])
    #intercept = mean[1:num_dim] - (tuple(mean[0] * x for x in slope))
    error = point_line_distance(data, slope, intercept)

    return Line(data, slope, intercept, mean, error)

def split_segment(segment):

    x1 = np.mean(segment.data[0])
    a = -np.divide(np.ones(x1.size),(segment.slope))

    y1 = np.multiply(segment.slope,x1) + segment.intercept
    b = y1 - np.multiply(a,x1)


    split1 = (segment.data[1] - (np.multiply(a, segment.data[0]) + b)) > 0
    split2 = split1 == False

    #split_point = np.int(np.floor(len(segment.data[0])/2))

    segment1 = line_of_best_fit(np.vstack((segment.data[0][split1[0]],segment.data[1][split1[0]])))
    segment2 = line_of_best_fit(np.vstack((segment.data[0][split2[0]],segment.data[1][split2[0]])))

    return (segment1, segment2)

def merge_segments(segment1, segment2, max_error):
    data = np.hstack((segment1.data,segment2.data))

    segment = line_of_best_fit(data)

    if segment.error < max_error:
        return segment
    else:
        return None

def split_and_merge(data, max_error):

    #For waveforms the pointwise error may be calculated as:
    #e_i = | y_i - p(x_i)
    # Where p(x_i) is the approximating polynomial

    #error(i) = abs(y(i) - polynomial_x(i))

    #For arbitrary plane curves, e_i is the Euclidean distance between
    #{x_i,y_i} and the approximating curve evaluated at that point.
    #For a line with equation sin(phi)*x + cos(phi)*y = d
    #e_i = |sin(phi)*x_i + cos(phi)*y_i - d|
    #where phi is the angle of the line with the x axis
    #where d is the distance of the line from the origin

    #We shall use the L2-norm
    #E_2 = sum(e_i^2)

    #Initialisation (Line of Best Fit)
    segments = []
    segments.append(line_of_best_fit(data))

    #Begin Split-Merge Algorithm
    terminate = False
    while(terminate == False):
        num_segments = len(segments)

        # Calculating Error (L2-distance from line for each segment)
        terminate = True
        step1 = True
        i = 0
        while(step1):
            # Step 1: For i = 1,2, n, check if Ei exceeds Emax. If
            # it does split the ith interval into two and increment nr.
            # The dividing point is determined by Rule A below. Calculate
            # the error norms on each new interval.
            if(segments[i].error > max_error):
                terminate = False
                # Rule A: If two or more points are known where the
                # pointwise error is maximum then use as a dividing point
                # the midpoint between a pair of them. Otherwise divide Si
                # in half.
                segments[i], segment2 = split_segment(segments[i])
                segments.insert(i+1, segment2)
                num_segments = num_segments + 1
            else:
                i = i+1
                if i == num_segments:
                    step1 = False
                #split set

        if(num_segments > 1):
            step2 = True
        else:
            step2 = False
        i = 0
        while(step2):
            # Step 2: For i = 1,2,... n, - 1 merge segments Si and
            # Si+, provided that this will result in a new segment with
            # Ei < Emax. Then decrease n, by one and calculate the
            # error norm on the new interval.

            # In Step 2, one can either try to merge all pairs of adjacent
            # segments and cancel the merge if Ei > Emax or he
            # can use Propositions 1 or 2 above to select pairs for which
            # a merge will be attempted.
            merged = merge_segments(segments[i], segments[i+1], max_error)
            if(merged is None):
                i=i+1
            else:
                terminate = False
                segments[i] = merged
                del segments[i+1]
                num_segments = num_segments - 1
            if i == (num_segments - 1):
                step2 = False

        if(num_segments > 1):
            step3 = True
        else:
            step3 = False
        i = 0
        flag = 0
        while(step3):
            # Step) 3: Do one iteration of algorithm R.
            if(segments[i].error > segments[i+1].error):
                #Check last point of i to i+1
                seg_error = point_line_distance(np.vstack((segments[i].data[0][:-2],segments[i].data[1][:-2])),
                                                           segments[i].slope, segments[i].intercept)
                point_error = point_line_distance(np.vstack((segments[i].data[0][-1],segments[i].data[1][-1])),
                                                    segments[i + 1].slope, segments[i + 1].intercept)
                # If shifted point would have less error than segments
                if(max(seg_error, point_error) < max(segments[i].error, segments[i+1].error)):
                    #Move it over
                    num_data = len(segments[i+1].data)

                    segments[i+1].data = np.append(np.vstack((segments[i].data[0][-1], segments[i ].data[1][-1])),
                                                 segments[i+1].data, axis = 1)
                    segments[i].data = np.delete(segments[i].data, 0, (len(segments[i].data)-1))
                    #del segments[i].data[-1]
                    # Update error
                    segments[i+1].error = ((segments[i+1].error * num_data) + point_error) / (num_data + 1)
                    segments[i].error = seg_error
                    terminate = False
            else:
                if(segments[i].error < segments[i+1].error):
                    #Check first point of i+1 to i
                    point_error = point_line_distance(np.vstack((segments[i+1].data[0][0],segments[i+1].data[1][0])),
                                                      segments[i].slope, segments[i].intercept)
                    seg_error = point_line_distance(np.vstack((segments[i+1].data[0][1:],segments[i+1].data[1][1:])),
                                                    segments[i+1].slope, segments[i+1].intercept)
                    # If shifted point would have less error than segments
                    if (max(point_error, seg_error) < max(segments[i].error, segments[i + 1].error)):
                        # Move it over
                        num_data = len(segments[i].data)
                        segments[i].data = np.append(segments[i].data, np.vstack((segments[i + 1].data[0][0], segments[i + 1].data[1][0])),
                                  axis=1)
                        segments[i + 1].data = np.delete(segments[i+1].data, 0, 1)
                        #Update error
                        segments[i].error = ((segments[i].error * num_data) + point_error)/(num_data+1)
                        segments[i + 1].error = seg_error
                        terminate = False

            i = i + 2
            if i >= (num_segments-1):
                if(flag == 0):
                    flag = 1
                    i = 1
                    if i >= (num_segments - 1):
                        step3 = False
                else:
                    step3 = False
        # Step 4: If no changes in the segments have occurred in
        # any of Steps 1-3 above, then terminate. Or else go to
        # Step 1.
        # In Step 4, it is usually necessary to check only if endpoint
        # adjustments have been made in Step 3 since this is
        # the last "active" step (see proof of theorem below).

    return segments
    #For the second case, the "split-and-merge" procedure
    #takes the following form.
    #Step la: If E > Emax then go to Step lb, or else go to
    #Step 2.
    #Step lb: Find the segment with the largest Ei and split
    #it into two as in Step 1 of the first case. Then go to Step la.
    #Step 2: Find the pair of adjacent segments whose merge
    #will cause the smallest increase in E and merge them if
    #this will not cause E to exceed Emax. Repeat till no further
    #mergers are possible.
    #Step 3: Do one iteration of algorithm R.
    #Step 4: If no changes in the segments have occurred in
    #Steps 2 or 3, then terminate. Or else go to Step 1.
    #It should be emphasized that the above procedures
    #have a dual goal. First they find a local minimum for the
    #number of segments so that the constraints on the error
    #norm(s) are satisfied. However, for that number of segments
    #there is a range of values for the error norm and
    #the procedure, for the types of R mentioned above, proceeds
    #next to minimize the latter. The minima are again
    #local.