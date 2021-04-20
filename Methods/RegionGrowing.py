import cv2
import numpy as np
import random
import sys


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

class RegionGrow:
    def __init__(self):
        pass

    def getGrayDiff(self, img, currentPoint, tmpPoint):
        return abs(int(img[currentPoint.y, currentPoint.x]) - int(img[tmpPoint.y, tmpPoint.x]))

    def selectConnects(self, p):
        if p != 0:
            connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), \
                        Point(0, 1), Point(-1, 1), Point(-1, 0)]
        else:
            connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return connects


    def regionGrowApply(self, img, seeds, thresh, p=1):
        height, width = img.shape
        seedMark = np.zeros(img.shape, dtype = np.uint8)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        label = 255
        connects = self.selectConnects(p)
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)

            seedMark[currentPoint.y, currentPoint.x] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= width or tmpY >= height:
                    continue
                grayDiff = self.getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                if grayDiff < thresh and seedMark[tmpY, tmpX] == 0:
                    seedMark[tmpY, tmpX] = 255
                    seedList.append(Point(tmpX, tmpY))
        #print(seedMark.shape)
        #cv2.imshow("seed",seedMark)
        #cv2.waitKey(0)
        seedMark = cv2.bitwise_not(seedMark)
        return seedMark

    def regionGrowLocalApply(self, img, seeds, diff_thre, binary_high_thre, binary_low_thre, size_thre, p=1):
        height, width = img.shape
        seedMark = np.zeros(img.shape, dtype = np.uint8)
        seedList = []
        for seed in seeds:
            seedList.append(seed)
        label = 255
        connects = self.selectConnects(p)
        size = 0
        while (len(seedList) > 0):
            currentPoint = seedList.pop(0)
            seedMark[currentPoint.y, currentPoint.x] = label
            for i in range(8):
                tmpX = currentPoint.x + connects[i].x
                tmpY = currentPoint.y + connects[i].y
                if tmpX < 0 or tmpY < 0 or tmpX >= width or tmpY >= height:
                    continue
                grayDiff = self.getGrayDiff(img, currentPoint, Point(tmpX, tmpY))
                grayOri = img[tmpY, tmpX]
                if grayDiff < diff_thre and seedMark[tmpY, tmpX] == 0 and grayOri < binary_high_thre and grayOri > binary_low_thre:
                    size += 1
                    seedMark[tmpY, tmpX] = 255
                    seedList.append(Point(tmpX, tmpY))
            if size > size_thre:
                break
        #print(seedMark.shape)
        #cv2.imshow("seed",seedMark)
        #cv2.waitKey(0)
        #seedMark = cv2.bitwise_not(seedMark)
        return seedMark

if __name__ == "__main__":
    img = cv2.imread('0.jpg', 0)
    seeds = [Point(200, 200)]
    RG = RegionGrow()
    binaryImg = RG.regionGrowApply(img, seeds, 15)
    cv2.imshow(' ', binaryImg)
    cv2.waitKey(0)