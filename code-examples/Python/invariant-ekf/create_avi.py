import cv2 as cv
import numpy as np

v_dir = 'riekf_localization_se2.avi'
fps = 20
img_t = cv.imread('./IMG/1.png')
img_size = (img_t.shape[1], img_t.shape[0])

N = 604

fourcc = cv.VideoWriter_fourcc('M', 'J', 'P', 'G')
v_w = cv.VideoWriter(v_dir, fourcc, fps, img_size)

for i in range(N):
    img = cv.imread('./IMG/{}.png'.format(i))
    v_w.write(img)

v_w.release()




