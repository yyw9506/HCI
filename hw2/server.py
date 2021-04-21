'''
You can modify the parameters, return values and data structures used in every function if it conflicts with your
coding style or you want to accelerate your code.

You can also import packages you want.

But please do not change the basic structure of this file including the function names. It is not recommended to merge
functions, otherwise it will be hard for TAs to grade your code. However, you can add helper function if necessary.

NAME : YUYAO WANG
ID : 112716165
DATE : 10/22/2020
HOMEWORK : HW2
'''

from flask import Flask, request
from flask import render_template
import time
import json
from scipy.interpolate import interp1d
import numpy as np
import math
from sklearn.metrics.pairwise import euclidean_distances

app = Flask(__name__)
print("CSE-518 HW2 (latest update: 2020-10-30)")
print("Flask App Name: ", app.name)

centroids_X = [50, 205, 135, 120, 100, 155, 190, 225, 275, 260, 295, 330, 275, 240, 310, 345, 30, 135, 85, 170, 240, 170, 65, 100, 205, 65]
centroids_Y = [85, 120, 120, 85, 50, 85, 85, 85, 50, 85, 85, 85, 120, 120, 50, 50, 50, 50, 85, 50, 50, 120, 50, 120, 50, 120]

words, probabilities = [], {}
template_points_X, template_points_Y = [], []
file = open('words_10000.txt')
content = file.read()
file.close()
content = content.split('\n')
for line in content:
    line = line.split('\t')
    words.append(line[0])
    probabilities[line[0]] = float(line[2])
    template_points_X.append([])
    template_points_Y.append([])
    for c in line[0]:
        template_points_X[-1].append(centroids_X[ord(c) - 97])
        template_points_Y[-1].append(centroids_Y[ord(c) - 97])

# generate sample points from 0 to 1 evenly
num_of_sampling = 100
pos_of_sampling = np.linspace(0, 1, num_of_sampling)

# alphas: the sum of all alphas is 1, lowest weight in the middle, rest of the points' weight increase linearly towards two ends
linear_factor_of_alphas = 2450 # (1125 * 2)
alphas = np.zeros((num_of_sampling))
mid_point = num_of_sampling // 2

for index in range(0, mid_point):
    x = index / linear_factor_of_alphas
    alphas[mid_point - index - 1], alphas[mid_point + index] = x, x

# calculate euclidean distance
def calculate_euclidean_distance(X1, Y1, X2, Y2):
    return math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2)

# calculate pairwise euclidean distance
def calculate_pairwise_distance(current_template_X, current_template_Y, gesture_sample_points_X, gesture_sample_points_Y):

    cumulative_distance = 0
    for index in range(0, 100):
        template_X = current_template_X[index]
        template_Y = current_template_Y[index]
        gesture_X = gesture_sample_points_X[index]
        gesture_Y = gesture_sample_points_Y[index]
        cumulative_distance += calculate_euclidean_distance(template_X, template_Y, gesture_X, gesture_Y)

    return cumulative_distance

def generate_sample_points(points_X, points_Y):
    '''Generate 100 sampled points for a gesture.

    In this function, we should convert every gesture or template to a set of 100 points, such that we can compare
    the input gesture and a template.

    :param points_X: A list of X-axis values of a gesture.
    :param points_Y: A list of Y-axis values of a gesture.

    :return:
        sample_points_X: A list of X-axis values of a gesture after sampling, containing 100 elements.
        sample_points_Y: A list of Y-axis values of a gesture after sampling, containing 100 elements.
    '''
    # TODO: Start sampling
    # get distance between points
    distance = np.sqrt(np.ediff1d(points_X, to_begin=0) ** 2 + np.ediff1d(points_Y, to_begin=0) ** 2)

    # get cumulative distance
    cumulative_distance = np.cumsum(distance)

    # avoid divide by zero
    if cumulative_distance[-1] == 0:
        cumulative_distance[-1] = 1

    # normalization
    cumulative_distance_norm = cumulative_distance / cumulative_distance[-1]

    # mapping 0-1 (normalized by cumulative distance) to X, Y
    X = interp1d(cumulative_distance_norm, points_X, kind='linear')
    Y = interp1d(cumulative_distance_norm, points_Y, kind='linear')

    # evenly sample X, Y from 0 - 1: pos_of_sampling[0, 0.01, 0.02, ..., 0.99, 1]
    return X(pos_of_sampling), Y(pos_of_sampling)

template_sample_points_X = []
template_sample_points_Y = []
for i in range(10000):
    X, Y = generate_sample_points(template_points_X[i], template_points_Y[i])
    template_sample_points_X.append(X)
    template_sample_points_Y.append(Y)

def do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y):
    '''Do pruning on the dictionary of 10000 words.

    In this function, we use the pruning method described in the paper (or any other method you consider reasonable)
    to narrow down the number of valid words so that ambiguity can be avoided.

    :param gesture_points_X: A list of X-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param gesture_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we have
        sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every template (10000 templates in total).
        Each of the elements is a 1D list and has the length of 100.

    :return:
        valid_words: A list of valid words after pruning.
        valid_probabilities: The corresponding probabilities of valid_words.
        valid_template_sample_points_X: 2D list, the corresponding X-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
        valid_template_sample_points_Y: 2D list, the corresponding Y-axis values of valid_words. Each of the elements
            is a 1D list and has the length of 100.
    '''
    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = [], [], []
    # TODO: Set your own pruning threshold
    threshold = 30 # Enter Value Here
    # TODO: Do pruning

    # get the start and end points of current gesture and all templates
    start_gesture_X, start_gesture_Y = gesture_points_X[0], gesture_points_Y[0]
    end_gesture_X, end_gesture_Y = gesture_points_X[-1], gesture_points_Y[-1]

    # calculate start and end distance
    for index in range(0, len(template_sample_points_X)):
        current_template_X = template_sample_points_X[index]
        current_template_Y = template_sample_points_Y[index]
        start_X = current_template_X[0]
        start_Y = current_template_Y[0]
        end_X = current_template_X[-1]
        end_Y = current_template_Y[-1]

        start_distance = calculate_euclidean_distance(start_X, start_Y, start_gesture_X, start_gesture_Y)
        end_distance = calculate_euclidean_distance(end_X, end_Y, end_gesture_X, end_gesture_Y)

        # add a candidate to valid_word only if the start distance + end distance are less than the threshold
        if start_distance + end_distance < threshold:
            valid_template_sample_points_X.append(current_template_X)
            valid_template_sample_points_Y.append(current_template_Y)
            valid_words.append(words[index])

    return valid_words, valid_template_sample_points_X, valid_template_sample_points_Y

def get_scaled_points(sample_points_X, sample_points_Y, L):
    x_maximum = max(sample_points_X)
    x_minimum = min(sample_points_X)
    W = x_maximum - x_minimum
    y_maximum = max(sample_points_Y)
    y_minimum = min(sample_points_Y)
    H = y_maximum - y_minimum

    # avoid divide by zero
    if max(H, W) != 0:
        r = L / max(H, W)
    else:
        r = L / 15

    gesture_X, gesture_Y = [], []
    for point_x, point_y in zip(sample_points_X, sample_points_Y):
        gesture_X.append(r * point_x)
        gesture_Y.append(r * point_y)

    centroid_x = (max(gesture_X) - min(gesture_X))/2
    centroid_y = (max(gesture_Y) - min(gesture_Y))/2
    scaled_X, scaled_Y = [], []
    for point_x, point_y in zip(gesture_X, gesture_Y):
        scaled_X.append(point_x - centroid_x)
        scaled_Y.append(point_y - centroid_y)

    return scaled_X, scaled_Y

def get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the shape score for every valid word after pruning.

    In this function, we should compare the sampled input gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a shape score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param valid_template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param valid_template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of shape scores.
    '''
    shape_scores = []
    # TODO: Set your own L
    L = 200 #Enter Value Here
    # TODO: Calculate shape scores

    # scale gesture sample points
    scaled_gesture_X, scaled_gesture_Y = get_scaled_points(gesture_sample_points_X, gesture_sample_points_Y, L)

    for index in range(0, len(valid_template_sample_points_X)):
        # scale valid template sample points
        scaled_current_template_X, scaled_current_template_Y = get_scaled_points(valid_template_sample_points_X[index], valid_template_sample_points_Y[index], L)

        # calculate pairwise distance
        pairwise_distance = calculate_pairwise_distance(scaled_current_template_X, scaled_current_template_Y, scaled_gesture_X, scaled_gesture_Y)
        shape_scores.append(pairwise_distance)

    return shape_scores

'''
Dropped begin
'''
# def get_small_d(p_X, p_Y, q_X, q_Y):
#     min_distance = []
#     for n in range(0, 100):
#         distance = math.sqrt((p_X - q_X[n])**2 + (p_Y - q_Y[n])**2)
#         min_distance.append(distance)
#     return (sorted(min_distance)[0])
#
# def get_big_d(p_X, p_Y, q_X, q_Y, r):
#     final_max = 0
#     for n in range(0, 100):
#         local_max = 0
#         distance = get_small_d(p_X[n], p_Y[n], q_X, q_Y)
#         local_max = max(distance-r , 0)
#         final_max += local_max
#     return final_max
#
# def get_delta(u_X, u_Y, t_X, t_Y, r, i):
#     D1 = get_big_d(u_X, u_Y, t_X, t_Y, r)
#     D2 = get_big_d(t_X, t_Y, u_X, u_Y, r)
#     if D1 == 0 and D2 == 0:
#         return 0
#     else:
#         return math.sqrt((u_X[i] - t_X[i])**2 + (u_Y[i] - t_Y[i])**2)
'''
Dropped end
'''

def get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y):
    '''Get the location score for every valid word after pruning.

    In this function, we should compare the sampled user gesture (containing 100 points) with every single valid
    template (containing 100 points) and give each of them a location score.

    :param gesture_sample_points_X: A list of X-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param gesture_sample_points_Y: A list of Y-axis values of input gesture points, which has 100 values since we
        have sampled 100 points.
    :param template_sample_points_X: 2D list, containing X-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.
    :param template_sample_points_Y: 2D list, containing Y-axis values of every valid template. Each of the
        elements is a 1D list and has the length of 100.

    :return:
        A list of location scores.
    '''
    location_scores = []
    radius = 15
    # TODO: Calculate location scores

    # initialize location scores to 0
    location_scores = np.zeros(len(valid_template_sample_points_X))

    # gesture points [[xi, yi]]
    gesture_points = [[gesture_sample_points_X[j], gesture_sample_points_Y[j]] for j in range(0, num_of_sampling)]

    for i in range(0, len(valid_template_sample_points_X)):
        # template points [[xi, yi]]
        template_points = [[valid_template_sample_points_X[i][j], valid_template_sample_points_Y[i][j]] for j in range(0, num_of_sampling)]
        # euclidean distance of each gesture point with each template point
        distances = euclidean_distances(gesture_points, template_points)
        # the distance of the closest gesture point to each template point
        template_gesture_min_distances = np.min(distances, axis=0)
        # the distance of the closest template point to each gesture point
        gesture_template_min_distances = np.min(distances, axis=1)
        # if any gesture point is not within the radius tunnel or any template point is not within the radius tunnel
        if np.any(gesture_template_min_distances > radius) or np.any(template_gesture_min_distances > radius):
            # delta is the distance of each gesture point with corresponding template point
            deltas = np.diagonal(distances)
            # location score = sum of product of alpha and delta for each point
            for index in range(0, len(deltas)):
                location_scores[i] += deltas[index] * alphas[index]

    return location_scores

def get_integration_scores(shape_scores, location_scores):
    integration_scores = []
    # TODO: Set your own shape weight
    shape_coef = 0.2 # Enter Value Here#
    # TODO: Set your own location weight
    location_coef = 0.8 # Enter Value Here#

    for i in range(len(shape_scores)):
        integration_scores.append(shape_coef * shape_scores[i] + location_coef * location_scores[i])
    return integration_scores

def get_best_word(valid_words, integration_scores):
    '''Get the best word.

    In this function, you should select top-n words with the highest integration scores and then use their corresponding
    probability (stored in variable "probabilities") as weight. The word with the highest weighted integration score is
    exactly the word we want.

    :param valid_words: A list of valid words.
    :param integration_scores: A list of corresponding integration scores of valid_words.
    :return: The most probable word suggested to the user.
    '''
    best_word = 'the'
    # TODO: Set your own range.
    n = 3 # Enter Value Here
    # TODO: Get the best word

    # get min score
    min_score = min(integration_scores)

    # get indexes of min score
    min_score_indices = np.where(integration_scores == min_score)[0]

    best_words = []
    for min_score_index in min_score_indices:
        best_words.append(valid_words[min_score_index])

        # only store top n words
        if len(best_words) >= n:
            break

    return " , ".join(best_words)

@app.route("/")
def init():
    return render_template('index.html')

@app.route('/shark2', methods=['POST'])
def shark2():

    start_time = time.time()
    data = json.loads(request.get_data())

    gesture_points_X = []
    gesture_points_Y = []

    for i in range(len(data)):
        gesture_points_X.append(data[i]['x'])
        gesture_points_Y.append(data[i]['y'])
    # gesture_points_X = [gesture_points_X]
    # gesture_points_Y = [gesture_points_Y]

    if len(gesture_points_X) <= 2:
        print("No gesture recognized.")
        return "No gesture recognized."

    gesture_sample_points_X, gesture_sample_points_Y = generate_sample_points(gesture_points_X, gesture_points_Y)

    valid_words, valid_template_sample_points_X, valid_template_sample_points_Y = do_pruning(gesture_points_X, gesture_points_Y, template_sample_points_X, template_sample_points_Y)

    best_word = "No word recognized."

    if len(valid_words) != 0:
        shape_scores = get_shape_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

        location_scores = get_location_scores(gesture_sample_points_X, gesture_sample_points_Y, valid_template_sample_points_X, valid_template_sample_points_Y)

        integration_scores = get_integration_scores(shape_scores, location_scores)

        best_word = get_best_word(valid_words, integration_scores)

    end_time = time.time()

    print('{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}')

    return '{"best_word": "' + best_word + '", "elapsed_time": "' + str(round((end_time - start_time) * 1000, 5)) + ' ms"}'

if __name__ == "__main__":
    app.run()