import numpy as np
import itertools
import math

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
    newID = itertools.count().next #Thread-Safe
    #https://stackoverflow.com/questions/1045344/how-do-you-create-an-incremental-id-in-a-python-class
    def __init__(self, data, slope, intercept, mean, error):
        # Note: If this item is deleted, ID will be released for reuse
        self.id = Line.newID()
        self.data = data
        self.slope = slope
        self.intercept = intercept
        self.mean = mean
        self.min = np.zeros(2)
        self.max = np.zeros(2)
        self.start = np.zeros(2)
        self.end = np.zeros(2)
        self.error = error
        self.start_connect = -1 #Depreciate
        self.end_connect = -1  #Depreciate
        self.nodes = [-1,-1]

class Node:
    newID = itertools.count().next  # Thread-Safe
    def __init__(self, coordinates):
        # Note: If this item is deleted, ID will be released for reuse
        self.id = Node.newID()
        self.coordinates = coordinates
        self.edges = []

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

def generate_line_ends(line):
    if (abs(line.slope[0][0]) < 1): #dY > dX
        x1 = np.min(line.data[0])
        x2 = np.max(line.data[0])
        if (line.slope[0] > 0): #Positive correlation
            y1 = int(math.floor(np.multiply(line.slope[0][0], x1) + line.intercept))
            y2 = int(math.ceil(np.multiply(line.slope[0][0], x2) + line.intercept))
        else: #Negative correlation
            y2 = int(math.ceil(np.multiply(line.slope[0][0], x1) + line.intercept))
            y1 = int(math.floor(np.multiply(line.slope[0][0], x2) + line.intercept))
    else: #dY < dX
        y1 = np.min(line.data[1])
        y2 = np.max(line.data[1])

        if (line.slope[0] > 0): #Positive Correlation
            x1 = int(math.floor(np.divide((y1 - line.intercept), line.slope[0][0])))
            x2 = int(math.ceil(np.divide((y2 - line.intercept), line.slope[0][0])))
        else: #Negative Correlation
            x2 = int(math.ceil(np.divide((y1 - line.intercept), line.slope[0][0])))
            x1 = int(math.floor(np.divide((y2 - line.intercept), line.slope[0][0])))


    # x1 -= np.minimum(x1,3)
    # x2 += np.minimum(xMax-x2,3)
    # y1 -= np.minimum(y1,3)
    # y2 += np.minimum(yMax-y2,3)

    line.min[0], line.min[1] = x1, y1
    line.max[0], line.max[1] = x2, y2
    if (line.slope[0] > 0):
        line.start[0], line.start[1] = x1, y1
        line.end[0], line.end[1] = x2, y2
    else:
        line.start[0], line.start[1] = x1, y2
        line.end[0], line.end[1] = x2, y1

def line_distances(line1,line2,big_number=9999999999999):
    distances = np.zeros(4)
        # start -> start
    if ((line1.start_connect < 0) & (line2.start_connect < 0)):
        distances[0] = point_distance(line1.start, line2.start)
    else:
        distances[0] = big_number
        # start -> end
    if ((line1.start_connect < 0) & (line2.end_connect < 0)):
        distances[1] = point_distance(line1.start, line2.end)
    else:
        distances[1] = big_number
        # end -> start
    if ((line1.end_connect < 0) & (line2.start_connect < 0)):
        distances[2] = point_distance(line1.end, line2.start)
    else:
        distances[2] = big_number
        # end -> end
    if ((line1.end_connect < 0) & (line2.start_connect < 0)):
        distances[3] = point_distance(line1.end, line2.end)
    else:
        distances[3] = big_number
    return distances

def point_distance(point1,point2):
    return np.sqrt(np.sum(np.power(point1 - point2, 2)))

def draw_line(start,end):
    data = [list(i) for i in zip(*[start,end])]
    line = line_of_best_fit(data)
    line.start = start
    line.end = end

    return line

def connect_lines(line1, line2, type):
    connect_switcher = {
        0: [line1.start, line2.start],
        1: [line1.start, line2.end],
        2: [line1.end, line2.start],
        3: [line1.end, line2.end],
    }

    [start, end] = connect_switcher[type]

    if(np.sum(abs(start - end)) == 0):
        attach_lines(line1, line2, type)
    else:
        new_line = draw_line(start, end)

        attach_switcher = {
            0: [0, 2], #l1(start) -> (start)new(end) -> (start)l2
            1: [0, 3], #l1(start) -> (start)new(end) -> (end)l2
            2: [2, 2], #l1(end) -> (start)new(end) -> (start)l2
            3: [2, 1], #l1(end) -> (start)new(end) -> (end)l2
        }
        [type1,type2] = attach_switcher[type]
        attach_lines(line1,new_line,type1) #start->start
        attach_lines(new_line, line2,type2) #end->start
    return new_line

def attach_lines(line1, line2, type):
    if(type == 0):
        node = Node(line1.start)
        line1.start_connect = line2.id
        line2.start_connect = line1.id
        line1.nodes[0] = node
        line2.nodes[0] = node
        node.edges.append(line1)
        node.edges.append(line2)
    else:
        if(type == 1):
            node = Node(line1.start)
            line1.start_connect = line2.id
            line2.end_connect = line1.id
            line1.nodes[0] = node
            line2.nodes[1] = node
            node.edges.append(line1)
            node.edges.append(line2)
        else:
            if(type == 2):
                node = Node(line1.end)
                line1.end_connect = line2.id
                line2.start_connect = line1.id
                line1.nodes[1] = node
                line2.nodes[0] = node
                node.edges.append(line1)
                node.edges.append(line2)
            else:
                if(type == 3):
                    node = Node(line1.end)
                    line1.end_connect = line2.id
                    line2.end_connect = line1.id
                    line1.nodes[1] = node
                    line2.nodes[1] = node
                    node.edges.append(line1)
                    node.edges.append(line2)
    return node


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