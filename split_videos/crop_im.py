import cv2
import os

path = "./pipeline_multi-larva-touching/"
save_path = "./pipeline_multi-larva-touching/cropped/"

SAVE_X_MIN = 100
SAVE_X_MAX = 380
SAVE_Y_MIN = 100
SAVE_Y_MAX = 380

im_name = os.listdir(path)
for im_n in im_name:
    if im_n[-3:] == "jpg":
        im = cv2.imread(path + im_n)
        im = cv2.flip(im, flipCode=-1)
        im = cv2.rotate(im, rotateCode=cv2.ROTATE_90_COUNTERCLOCKWISE)
        cv2.imwrite(save_path + im_n, im[SAVE_Y_MIN:SAVE_Y_MAX, SAVE_X_MIN:SAVE_X_MAX])