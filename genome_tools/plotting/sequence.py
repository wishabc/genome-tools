import re
from collections import OrderedDict

import numpy as np

from descartes.patch import PolygonPatch

from shapely.wkt import loads as load_wkt
from shapely import affinity

# -----------------------------------------------------------------------
#
#
# Code adoped from https://github.com/kundajelab/dragonn, (c) 2016 Kundaje Lab

def standardize_polygons_str(data_str):
    """Given a POLYGON string, standardize the coordinates to a 1x1 grid.
    Input : data_str (taken from above)
    Output: tuple of polygon objects
    """
    # find all of the polygons in the letter (for instance an A
    # needs to be constructed from 2 polygons)
    path_strs = re.findall("\(\(([^\)]+?)\)\)", data_str.strip())

    # convert the data into a numpy array
    polygons_data = []
    for path_str in path_strs:
        data = np.array([
            tuple(map(float, x.split())) for x in path_str.strip().split(",")])
        polygons_data.append(data)

    # standardize the coordinates
    min_coords = np.vstack([data.min(0) for data in polygons_data]).min(0)
    max_coords = np.vstack([data.max(0) for data in polygons_data]).max(0)
    for data in polygons_data:
        data[:, ] -= min_coords
        data[:, ] /= (max_coords - min_coords)

    polygons = []
    for data in polygons_data:
        polygons.append(load_wkt(
            "POLYGON((%s))" % ",".join(" ".join(map(str, x)) for x in data)))

    return tuple(polygons)

# ------------------------

import string

# Geometry taken from JTS TestBuilder Monospace font with fixed precision model
# of 1000.0

A = """MULTIPOLYGON (
((30.0781 64.2031, 19.6719 26.9062, 40.4844 26.9062, 30.0781 64.2031)),
((24.125 72.9062, 36.0781 72.9062, 58.4062 0, 48.1875 0, 42.8281 19, 17.2812 19, 12.0156 0, 1.8125 0, 24.125 72.9062))
)"""

A = """MULTIPOLYGON (
((24.7631 57.3346, 34.3963 57.3346, 52.391 -1.422, 44.1555 -1.422, 39.8363
  13.8905, 19.2476 13.8905, 15.0039 -1.422, 6.781 -1.422, 24.7631 57.3346)),
((29.5608 50.3205, 21.1742 20.2623, 37.9474 20.2623, 29.5608 50.3205))
)"""

B = """MULTIPOLYGON (((18.0156 34.8125, 18.0156 8.1094, 29.6875 8.1094, 33.6763 8.2969, 37.0488 8.8594, 39.8052 9.7969, 41.9453 11.1094, 43.5483 12.8525, 44.6934 15.082, 45.3804 17.7979, 45.6094 21, 45.3682 24.3262, 44.6445 27.1797, 43.4385 29.5605, 41.75 31.4688, 39.5488 32.9316, 36.8047 33.9766, 33.5176 34.6035, 29.6875 34.8125, 18.0156 34.8125), 
  (18.0156 64.7969, 18.0156 42.8281, 29.5 42.8281, 32.8169 42.999, 35.6426 43.5117, 37.9771 44.3662, 39.8203 45.5625, 41.2183 47.1396, 42.2168 49.1367, 42.8159 51.5537, 43.0156 54.3906, 42.8188 56.9565, 42.2285 59.1387, 41.2446 60.937, 39.8672 62.3516, 38.0386 63.4214, 35.7012 64.1855, 32.855 64.644, 29.5 64.7969, 18.0156 64.7969)), 
  ((8.1094 72.9062, 29.6875 72.9062, 34.9604 72.604, 39.5918 71.6973, 43.5815 70.186, 46.9297 68.0703, 49.5786 65.4019, 51.4707 62.2324, 52.606 58.562, 52.9844 54.3906, 52.7842 51.2236, 52.1836 48.4102, 51.1826 45.9502, 49.7812 43.8438, 47.9805 42.0947, 45.7812 40.707, 43.1836 39.6807, 40.1875 39.0156, 43.5801 38.228, 46.5859 36.9434, 49.2051 35.1616, 51.4375 32.8828, 53.2217 30.1831, 54.4961 27.1387, 55.2607 23.7495, 55.5156 20.0156, 55.1099 15.3657, 53.8926 11.3223, 52.9796 9.528, 51.8638 7.8853, 50.545 6.3942, 49.0234 5.0547, 45.3804 2.8433, 40.9434 1.2637, 35.7124 0.3159, 29.6875 0, 8.1094 0, 8.1094 72.9062)))"""

C = """POLYGON ((52.3906 2.5938, 48.5879 0.8418, 44.6797 -0.4141, 40.5996 -1.1699, 36.2812 -1.4219, 32.8754 -1.267, 29.6655 -0.8022, 26.6517 -0.0277, 23.834 1.0566, 21.2123 2.4508, 18.7866 4.1548, 16.557 6.1686, 14.5234 8.4922, 12.7089 11.0966, 11.1362 13.9526, 9.8055 17.0604, 8.7168 20.4199, 7.87 24.0311, 7.2651 27.894, 6.9022 32.0087, 6.7812 36.375, 6.903 40.7205, 7.2681 44.8193, 7.8766 48.6716, 8.7285 52.2773, 9.8239 55.6365, 11.1626 58.749, 12.7448 61.615, 14.5703 64.2344, 16.6134 66.5745, 18.8481 68.6025, 21.2745 70.3186, 23.8926 71.7227, 26.7023 72.8147, 29.7036 73.5947, 32.8966 74.0627, 36.2812 74.2188, 40.5996 73.9688, 44.6797 73.2188, 48.5879 71.9688, 52.3906 70.2188, 52.3906 60.1094, 48.6465 62.7383, 44.6328 64.6562, 40.4707 65.8281, 36.2812 66.2188, 31.7715 65.7549, 29.7437 65.175, 27.8672 64.3633, 26.1421 63.3196, 24.5684 62.0439, 23.146 60.5364, 21.875 58.7969, 19.7832 54.6123, 18.2891 49.4805, 17.3926 43.4014, 17.0938 36.375, 17.3926 29.376, 18.2891 23.3164, 19.7832 18.1963, 21.875 14.0156, 23.146 12.2761, 24.5684 10.7686, 26.1421 9.4929, 27.8672 8.4492, 29.7437 7.6375, 31.7715 7.0576, 36.2812 6.5938, 40.5352 6.9844, 44.7031 8.1562, 48.6875 10.0742, 52.3906 12.7031, 52.3906 2.5938))"""

D = """MULTIPOLYGON (
((21.4844 72.9062, 25.5227 72.7673, 29.2861 72.3506, 32.7747 71.656, 35.9883 70.6836, 38.927 69.4333, 41.5908 67.9053, 43.9797 66.0994, 46.0938 64.0156, 47.9468 61.6389, 49.5527 58.9541, 50.9116 55.9612, 52.0234 52.6602, 52.8882 49.051, 53.5059 45.1338, 53.8765 40.9084, 54 36.375, 53.8765 31.864, 53.5059 27.6592, 52.8882 23.7605, 52.0234 20.168, 50.9116 16.8816, 49.5527 13.9014, 47.9468 11.2273, 46.0938 8.8594, 43.9797 6.783, 41.5908 4.9834, 38.927 3.4607, 35.9883 2.2148, 32.7747 1.2458, 29.2861 0.5537, 25.5227 0.1384, 21.4844 0, 6.6875 0, 6.6875 72.9062, 21.4844 72.9062)),
((21.2969 8.1094, 27.0469 8.4922, 31.8594 9.6406, 33.9141 10.502, 35.7344 11.5547, 37.3203 12.7988, 38.6719 14.2344, 39.8291 15.9202, 40.832 17.915, 42.375 22.832, 43.3008 28.9854, 43.6094 36.375, 43.3027 43.833, 42.3828 50.0352, 40.8496 54.9814, 39.853 56.9836, 38.7031 58.6719, 37.3579 60.1074, 35.7754 61.3516, 33.9556 62.4043, 31.8984 63.2656, 27.0723 64.4141, 21.2969 64.7969, 16.6094 64.7969, 16.6094 8.1094, 21.2969 8.1094)))"""

E = """POLYGON ((9.625 72.9062, 52.875 72.9062, 52.875 64.5938, 19.4844 64.5938, 19.4844 43.0156, 51.4219 43.0156, 51.4219 34.7188, 19.4844 34.7188, 19.4844 8.2969, 53.8125 8.2969, 53.8125 0, 9.625 0, 9.625 72.9062))"""

F = """POLYGON ((11.375 72.9062, 54.2969 72.9062, 54.2969 64.5938, 21.2969 64.5938, 21.2969 43.1094, 51.2188 43.1094, 51.2188 34.8125, 21.2969 34.8125, 21.2969 0, 11.375 0, 11.375 72.9062))"""

G = """POLYGON ((53.9062 6, 51.8672 4.2827, 49.7031 2.7871, 47.4141 1.5132, 45 0.4609, 42.4727 -0.3628, 39.8438 -0.9512, 34.2812 -1.4219, 30.9088 -1.2666, 27.729 -0.8008, 24.7418 -0.0244, 21.9473 1.0625, 19.3453 2.46, 16.936 4.168, 14.7194 6.1865, 12.6953 8.5156, 10.8881 11.1248, 9.3218 13.9834, 7.9965 17.0916, 6.9121 20.4492, 6.0687 24.0564, 5.4663 27.9131, 5.1049 32.0193, 4.9844 36.375, 5.1064 40.7205, 5.4727 44.8193, 6.083 48.6716, 6.9375 52.2773, 8.0361 55.6365, 9.3789 58.749, 10.9658 61.615, 12.7969 64.2344, 14.8452 66.5745, 17.084 68.6025, 19.5132 70.3186, 22.1328 71.7227, 24.9429 72.8147, 27.9434 73.5947, 31.1343 74.0627, 34.5156 74.2188, 39.0117 73.8945, 43.3125 72.9219, 47.4414 71.293, 51.4219 69, 51.4219 58.8906, 47.418 62.1504, 43.3125 64.4297, 39.0352 65.7715, 34.5156 66.2188, 30.0107 65.7529, 27.9832 65.1707, 26.1055 64.3555, 24.3777 63.3074, 22.7998 62.0264, 21.3718 60.5125, 20.0938 58.7656, 17.9883 54.5713, 16.4844 49.4414, 15.582 43.376, 15.2812 36.375, 15.5728 29.2744, 16.4473 23.1602, 17.9048 18.0322, 19.9453 13.8906, 21.1896 12.1804, 22.5903 10.6982, 24.1476 9.4441, 25.8613 8.418, 27.7316 7.6199, 29.7583 7.0498, 34.2812 6.5938, 37.3477 6.7832, 40.0156 7.3516, 42.3672 8.3223, 44.4844 9.7188, 44.4844 29.2969, 33.8906 29.2969, 33.8906 37.4062, 53.9062 37.4062, 53.9062 6))"""

H = """POLYGON ((6.6875 72.9062, 16.6094 72.9062, 16.6094 43.0156, 43.6094 43.0156, 43.6094 72.9062, 53.5156 72.9062, 53.5156 0, 43.6094 0, 43.6094 34.7188, 16.6094 34.7188, 16.6094 0, 6.6875 0, 6.6875 72.9062))"""

I = """POLYGON ((9.8125 72.9062, 50.2969 72.9062, 50.2969 64.5938, 35.0156 64.5938, 35.0156 8.2969, 50.2969 8.2969, 50.2969 0, 9.8125 0, 9.8125 8.2969, 25.0938 8.2969, 25.0938 64.5938, 9.8125 64.5938, 9.8125 72.9062))"""


J = """POLYGON ((5.3281 2.9844, 5.3281 14.5, 9.8398 11.041, 14.5 8.5703, 19.3203 7.0879, 21.7949 6.7173, 24.3125 6.5938, 27.5405 6.8213, 30.2402 7.5039, 32.4116 8.6416, 34.0547 10.2344, 35.2612 12.4463, 36.123 15.4414, 36.6401 19.2197, 36.8125 23.7812, 36.8125 64.5938, 18.2188 64.5938, 18.2188 72.9062, 46.6875 72.9062, 46.6875 23.7812, 46.3652 17.4014, 45.3984 12.0273, 43.7871 7.6592, 42.7397 5.8523, 41.5312 4.2969, 40.1343 2.9565, 38.5215 1.7949, 36.6929 0.812, 34.6484 0.0078, 29.9121 -1.0645, 24.3125 -1.4219, 19.6387 -1.1523, 14.9922 -0.3438, 10.2598 1.0273, 5.3281 2.9844))"""

K = """POLYGON ((6.6875 72.9062, 16.6094 72.9062, 16.6094 40.4844, 47.4062 72.9062, 58.9844 72.9062, 30.6094 43.1094, 59.8125 0, 47.9062 0, 24.125 36.5312, 16.6094 28.5156, 16.6094 0, 6.6875 0, 6.6875 72.9062))"""

L = """POLYGON ((10.5 72.9062, 20.4062 72.9062, 20.4062 8.2969, 55.6094 8.2969, 55.6094 0, 10.5 0, 10.5 72.9062))"""

M = """POLYGON ((4.2031 72.9062, 17.3906 72.9062, 29.9844 35.7969, 42.6719 72.9062, 55.9062 72.9062, 55.9062 0, 46.7812 0, 46.7812 64.4062, 33.7969 25.9844, 26.3125 25.9844, 13.2812 64.4062, 13.2812 0, 4.2031 0, 4.2031 72.9062))"""

N = """POLYGON ((6.7812 72.9062, 19.2812 72.9062, 43.8906 12.8906, 43.8906 72.9062, 53.4219 72.9062, 53.4219 0, 40.9219 0, 16.3125 60.0156, 16.3125 0, 6.7812 0, 6.7812 72.9062))"""

O = """MULTIPOLYGON (
((44.1875 36.375, 43.9814 43.833, 43.3633 50.1445, 42.333 55.3096, 40.8906 59.3281, 38.9785 62.3428, 36.5391 64.4961, 35.1216 65.2498, 33.5723 65.7881, 30.0781 66.2188, 26.6045 65.7881, 23.6523 64.4961, 21.2217 62.3428, 19.3125 59.3281, 17.8701 55.3096, 16.8398 50.1445, 16.2217 43.833, 16.0156 36.375, 16.2217 28.938, 16.8398 22.6426, 17.8701 17.4888, 19.3125 13.4766, 21.2217 10.4653, 23.6523 8.3145, 26.6045 7.0239, 30.0781 6.5938, 33.5723 7.0225, 36.5391 8.3086, 38.9785 10.4521, 40.8906 13.4531, 42.333 17.458, 43.3633 22.6133, 43.9814 28.9189, 44.1875 36.375)), 
  ((54.5 36.375, 54.123 27.4561, 52.9922 19.7461, 52.144 16.3445, 51.1074 13.2451, 49.8823 10.448, 48.4688 7.9531, 46.8621 5.7559, 45.0576 3.8516, 43.0554 2.2402, 40.8555 0.9219, 38.4578 -0.1035, 35.8623 -0.8359, 33.0691 -1.2754, 30.0781 -1.4219, 27.0876 -1.2761, 24.2959 -0.8389, 21.7029 -0.1101, 19.3086 0.9102, 17.113 2.2219, 15.1162 3.8252, 13.3181 5.72, 11.7188 7.9062, 10.3125 10.3916, 9.0938 13.1836, 8.0625 16.2822, 7.2188 19.6875, 6.0938 27.418, 5.7188 36.375, 6.0957 45.3145, 7.2266 53.0391, 8.0747 56.4458, 9.1113 59.5488, 10.3364 62.3481, 11.75 64.8438, 13.3557 67.041, 15.1572 68.9453, 17.1545 70.5566, 19.3477 71.875, 21.7366 72.9004, 24.3213 73.6328, 27.1018 74.0723, 30.0781 74.2188, 33.0691 74.0723, 35.8623 73.6328, 38.4578 72.9004, 40.8555 71.875, 43.0554 70.5566, 45.0576 68.9453, 46.8621 67.041, 48.4688 64.8438, 49.8823 62.3481, 51.1074 59.5488, 52.144 56.4458, 52.9922 53.0391, 54.123 45.3145, 54.5 36.375)))"""

P = """MULTIPOLYGON (
((9.625 72.9062, 30.9062 72.9062, 36.6392 72.5596, 41.6348 71.5195, 45.8931 69.7861, 49.4141 67.3594, 50.8917 65.8923, 52.1724 64.2646, 54.1426 60.5273, 55.3247 56.1475, 55.7188 51.125, 55.3267 46.0566, 54.1504 41.6484, 53.2682 39.6919, 52.1899 37.9004, 50.9156 36.2739, 49.4453 34.8125, 45.9341 32.3994, 41.6738 30.6758, 36.6646 29.6416, 30.9062 29.2969, 19.4844 29.2969, 19.4844 0, 9.625 0, 9.625 72.9062)),
((19.4844 64.7969, 19.4844 37.4062, 30.9062 37.4062, 34.1396 37.6318, 36.9961 38.3086, 39.4756 39.4365, 41.5781 41.0156, 43.2529 43.001, 44.4492 45.3477, 45.167 48.0557, 45.4062 51.125, 45.1685 54.1929, 44.4551 56.8965, 43.2661 59.2358, 41.6016 61.2109, 39.5063 62.7798, 37.0254 63.9004, 34.1587 64.5728, 30.9062 64.7969, 19.4844 64.7969))
)"""


Q = """MULTIPOLYGON (
((31.9844 -1.3125, 31.0078 -1.3672, 29.9844 -1.4219, 27.03 -1.2754, 24.2686 -0.8359, 21.7 -0.1035, 19.3242 0.9219, 17.1414 2.2402, 15.1514 3.8516, 13.3542 5.7559, 11.75 7.9531, 10.3364 10.448, 9.1113 13.2451, 8.0747 16.3445, 7.2266 19.7461, 6.0957 27.4561, 5.7188 36.375, 6.0957 45.3145, 7.2266 53.0391, 8.0747 56.4458, 9.1113 59.5488, 10.3364 62.3481, 11.75 64.8438, 13.3557 67.041, 15.1572 68.9453, 17.1545 70.5566, 19.3477 71.875, 21.7366 72.9004, 24.3213 73.6328, 27.1018 74.0723, 30.0781 74.2188, 33.0691 74.0723, 35.8623 73.6328, 38.4578 72.9004, 40.8555 71.875, 43.0554 70.5566, 45.0576 68.9453, 46.8621 67.041, 48.4688 64.8438, 49.8823 62.3481, 51.1074 59.5488, 52.144 56.4458, 52.9922 53.0391, 54.123 45.3145, 54.5 36.375, 54.2905 29.5454, 53.6621 23.416, 52.6147 17.9868, 51.1484 13.2578, 49.2583 9.2065, 46.9395 5.8105, 44.1919 3.0698, 41.0156 0.9844, 50.7812 -8.2969, 43.4062 -13.1875, 31.9844 -1.3125)), 
  ((44.1875 36.375, 43.9814 43.833, 43.3633 50.1445, 42.333 55.3096, 40.8906 59.3281, 38.9785 62.3428, 36.5391 64.4961, 35.1216 65.2498, 33.5723 65.7881, 30.0781 66.2188, 26.6045 65.7881, 23.6523 64.4961, 21.2217 62.3428, 19.3125 59.3281, 17.8701 55.3096, 16.8398 50.1445, 16.2217 43.833, 16.0156 36.375, 16.2217 28.938, 16.8398 22.6426, 17.8701 17.4888, 19.3125 13.4766, 21.2217 10.4653, 23.6523 8.3145, 26.6045 7.0239, 30.0781 6.5938, 33.5723 7.0225, 36.5391 8.3086, 38.9785 10.4521, 40.8906 13.4531, 42.333 17.458, 43.3633 22.6133, 43.9814 28.9189, 44.1875 36.375)))"""

R = """MULTIPOLYGON (
((37.1094 34.4219, 38.9453 33.8286, 40.6406 33.0176, 42.1953 31.9888, 43.6094 30.7422, 45.0352 29.0493, 46.625 26.6816, 50.2969 19.9219, 60.2031 0, 49.6094 0, 40.9219 18.4062, 39.0869 21.9868, 37.3477 24.8691, 35.7041 27.0532, 34.1562 28.5391, 32.541 29.5337, 30.6953 30.2441, 28.6191 30.6704, 26.3125 30.8125, 16.8906 30.8125, 16.8906 0, 6.9844 0, 6.9844 72.9062, 27.2969 72.9062, 32.9531 72.5674, 37.9062 71.5508, 42.1562 69.8564, 45.7031 67.4844, 48.499 64.4717, 50.4961 60.8555, 51.6943 56.6357, 52.0938 51.8125, 51.8481 48.3623, 51.1113 45.2461, 49.8833 42.4639, 48.1641 40.0156, 45.9995 37.9551, 43.4355 36.3359, 40.4722 35.1582, 37.1094 34.4219)), 
  ((16.8906 64.7969, 16.8906 38.9219, 27.6875 38.9219, 31.001 39.1201, 33.8633 39.7148, 36.2744 40.7061, 38.2344 42.0938, 39.752 43.8906, 40.8359 46.1094, 41.4863 48.75, 41.7031 51.8125, 41.4727 54.7764, 40.7812 57.3711, 39.6289 59.5967, 38.0156 61.4531, 35.9629 62.916, 33.4922 63.9609, 30.6035 64.5879, 27.2969 64.7969, 16.8906 64.7969)))"""

S = """POLYGON ((49.4219 70.4062, 49.4219 60.4062, 44.918 62.9297, 40.4062 64.75, 35.8711 65.8516, 31.2969 66.2188, 27.9883 66.0156, 25.0469 65.4062, 22.4727 64.3906, 20.2656 62.9688, 18.4951 61.2021, 17.2305 59.1523, 16.4717 56.8193, 16.2188 54.2031, 16.3848 51.9253, 16.8828 49.9355, 17.7129 48.2339, 18.875 46.8203, 20.4922 45.603, 22.6875 44.4902, 25.4609 43.4819, 28.8125 42.5781, 33.9844 41.4062, 38.7617 40.04, 42.8438 38.332, 46.2305 36.2822, 48.9219 33.8906, 50.9727 31.0957, 52.4375 27.8359, 53.3164 24.1113, 53.6094 19.9219, 53.2065 15.0273, 51.998 10.75, 51.0917 8.8428, 49.9839 7.0898, 48.6747 5.4912, 47.1641 4.0469, 43.5757 1.6543, 39.2559 -0.0547, 34.2046 -1.0801, 28.4219 -1.4219, 23.2832 -1.1465, 18.1172 -0.3203, 12.9277 1.0566, 7.7188 2.9844, 7.7188 13.4844, 13.1777 10.3867, 18.3359 8.25, 23.3613 7.0078, 28.4219 6.5938, 31.9487 6.8027, 35.0605 7.4297, 37.7573 8.4746, 40.0391 9.9375, 41.854 11.7754, 43.1504 13.9453, 43.9282 16.4473, 44.1875 19.2812, 44.0093 21.854, 43.4746 24.1035, 42.5835 26.0298, 41.3359 27.6328, 39.645 28.9917, 37.4238 30.1855, 34.6724 31.2144, 31.3906 32.0781, 26.125 33.2969, 21.3945 34.5918, 17.3594 36.1797, 14.0195 38.0605, 11.375 40.2344, 9.3652 42.7529, 7.9297 45.668, 7.0684 48.9795, 6.7812 52.6875, 7.1919 57.3276, 8.4238 61.4824, 10.4771 65.1519, 13.3516 68.3359, 16.9067 70.9097, 21.002 72.748, 25.6372 73.8511, 30.8125 74.2188, 35.123 73.9805, 39.6484 73.2656, 44.4082 72.0742, 49.4219 70.4062))"""

T = """POLYGON ((2.2969 72.9062, 57.9062 72.9062, 57.9062 64.5938, 35.1094 64.5938, 35.1094 0, 25.2031 0, 25.2031 64.5938, 2.2969 64.5938, 2.2969 72.9062))"""

U = """POLYGON ((7.1719 27.9844, 7.1719 72.9062, 17.0938 72.9062, 17.0938 23.4844, 17.3828 15.8984, 17.7832 13.8887, 18.4062 12.4062, 20.3164 9.873, 22.9219 8.0547, 26.1875 6.959, 30.0781 6.5938, 33.998 6.959, 37.2578 8.0547, 39.8574 9.873, 41.7969 12.4062, 42.4199 13.8809, 42.8203 15.8672, 43.1094 23.3906, 43.1094 72.9062, 52.9844 72.9062, 52.9844 27.9844, 52.6367 18.416, 52.2021 14.8462, 51.5938 12.0859, 50.7715 9.8325, 49.6953 7.7832, 48.3652 5.938, 46.7812 4.2969, 43.3281 1.7852, 39.4062 0, 34.9961 -1.0664, 30.0781 -1.4219, 25.1973 -1.0664, 20.8047 0, 16.873 1.7852, 13.375 4.2969, 11.8164 5.9233, 10.5 7.7715, 9.4258 9.8413, 8.5938 12.1328, 7.9717 14.9253, 7.5273 18.498, 7.1719 27.9844))"""

V = """POLYGON ((30.0781 8.2969, 47.2188 72.9062, 57.4219 72.9062, 36.0781 0, 24.125 0, 2.7812 72.9062, 12.9844 72.9062, 30.0781 8.2969))"""

W = """POLYGON ((0 72.9062, 9.625 72.9062, 16.6094 13.7188, 24.9062 52.875, 35.2031 52.875, 43.6094 13.625, 50.5938 72.9062, 60.2031 72.9062, 49.3125 0, 39.9844 0, 30.0781 43.3125, 20.2188 0, 10.8906 0, 0 72.9062))"""

X = """POLYGON ((4.2031 72.9062, 14.7969 72.9062, 30.8125 45.4062, 47.125 72.9062, 57.7188 72.9062, 35.8906 38.625, 59.2812 0, 48.6875 0, 30.8125 31.3906, 11.5312 0, 0.875 0, 25.2969 38.625, 4.2031 72.9062))"""

Y = """POLYGON ((1.8125 72.9062, 12.3125 72.9062, 30.0781 40.7188, 47.7969 72.9062, 58.4062 72.9062, 35.0156 32.7188, 35.0156 0, 25.0938 0, 25.0938 32.7188, 1.8125 72.9062))"""

Z = """POLYGON ((8.6875 72.9062, 56 72.9062, 56 65.375, 17.9219 8.2969, 57.0781 8.2969, 57.0781 0, 7.625 0, 7.625 7.5156, 44.6719 64.5938, 8.6875 64.5938, 8.6875 72.9062))"""


all_letters = {l: globals()[l] for l in string.ascii_uppercase}

# ----------------------

DNA = ["A", "C", "G", "T"]
RNA = ["A", "C", "G", "U"]
AMINO_ACIDS = ["A", "R", "N", "D", "B", "C", "E", "Q", "Z", "G", "H",
               "I", "L", "K", "M", "F", "P", "S", "T", "W", "Y", "V"]

letter_polygons = {k: standardize_polygons_str(v) for k, v in all_letters.items()}

VOCABS = {"DNA": OrderedDict([("A", "green"),
                              ("C", "blue"),
                              ("G", "orange"),
                              ("T", "red")]),
          "RNA": OrderedDict([("A", "green"),
                              ("C", "blue"),
                              ("G", "orange"),
                              ("U", "red")]),
          "AA": OrderedDict([('A', '#CCFF00'),
                             ('B', "orange"),
                             ('C', '#FFFF00'),
                             ('D', '#FF0000'),
                             ('E', '#FF0066'),
                             ('F', '#00FF66'),
                             ('G', '#FF9900'),
                             ('H', '#0066FF'),
                             ('I', '#66FF00'),
                             ('K', '#6600FF'),
                             ('L', '#33FF00'),
                             ('M', '#00FF00'),
                             ('N', '#CC00FF'),
                             ('P', '#FFCC00'),
                             ('Q', '#FF00CC'),
                             ('R', '#0000FF'),
                             ('S', '#FF3300'),
                             ('T', '#FF6600'),
                             ('V', '#99FF00'),
                             ('W', '#00CCFF'),
                             ('Y', '#00FFCC'),
                             ('Z', 'blue')]),
          "RNAStruct": OrderedDict([("P", "red"),
                                    ("H", "green"),
                                    ("I", "blue"),
                                    ("M", "orange"),
                                    ("E", "violet")]),
          }
# make sure things are in order
VOCABS["AA"] = OrderedDict((k, VOCABS["AA"][k]) for k in AMINO_ACIDS)
VOCABS["DNA"] = OrderedDict((k, VOCABS["DNA"][k]) for k in DNA)
VOCABS["RNA"] = OrderedDict((k, VOCABS["RNA"][k]) for k in RNA)

# ------------------------

def add_letter_to_axis(ax, let, col, x, y, height):
    """Add 'let' with position x,y and height height to matplotlib axis 'ax'.
    """
    if len(let) == 2:
        colors = [col, "white"]
    elif len(let) == 1:
        colors = [col]
    else:
        raise ValueError("3 or more Polygons are not supported")

    for polygon, color in zip(let, colors):
        new_polygon = affinity.scale(
            polygon, yfact=height, origin=(0, 0, 0))
        new_polygon = affinity.translate(
            new_polygon, xoff=x, yoff=y)
        patch = PolygonPatch(
            new_polygon, edgecolor=color, facecolor=color)
        ax.add_patch(patch)
    return
