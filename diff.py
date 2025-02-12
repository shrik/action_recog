frame_1 = open("frame_1.txt", "r").read()
frame_1_local = open("frame_1_local.txt", "r").read()

frame_1 = [float(i) for i in frame_1.split(" ")]
frame_1_local = [float(i) for i in frame_1_local.split(" ")]

import numpy as np
def to_nparray(data):
    return np.array(data).reshape(1280, 7 , 7)

print(len(frame_1))
print(len(frame_1_local))

frame_1 = to_nparray(frame_1)
frame_1_local = to_nparray(frame_1_local)

print(np.sum(np.abs(frame_1 - frame_1_local)))
print(np.sum(frame_1) - np.sum(frame_1_local))
