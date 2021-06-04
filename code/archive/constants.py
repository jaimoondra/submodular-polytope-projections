import math

# Constants:
n = 100  # size of ground set
base_decimal_accuracy = 5  # All inputs and outputs will have this accuracy
minimum_decimal_difference = math.pow(10, -base_decimal_accuracy)  # for float accuracy
high_decimal_accuracy = 12  # Used only by inner computations in algorithms
dist_square = 50  # todo: add description
std_dev_point = 5  # Standard deviation for selection on random point
seeds = [1234, 90, 0, 6454, 6, 444444, 39256, 7527, 50604, 24743, 47208, 28212, 19019, 41225, 23406,
         52847, 62727, 3034, 55949, 13206, 8086, 55396, 21709, 10223, 41131, 45982, 51335, 19036,
         9056, 17681, 15141, 6306, 63724, 42770, 35394, 44056, 22564, 50203, 13494, 2617, 62882,
         35918, 2597, 43039, 7228, 35110, 63328, 35294, 21347, 69, 55129, 64711, 24826, 25899,
         13623, 64414, 18845, 51362, 15405, 39271, 29175, 31418, 3071, 9840, 49312, 63306, 48069,
         48216, 59896, 52064, 7533, 9390, 36907, 25146, 7840, 42243, 35634, 50032, 12157, 47424,
         39071, 9496, 30727, 11739, 60247, 33845, 25754, 45533, 27374, 29006, 3133, 8072, 6823,
         55874, 54767, 29723, 50573, 19110, 40861, 17731, 20386, 54415, 11486, 63471, 26744, 3881]
# Multiple seeds to ensure both randomness and repeatability
