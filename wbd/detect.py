import numpy as np
import cv2 as cv
import cv2.aruco as aruco


def detectBoardOtsu(wdata):
    _, thresh = cv.threshold(wdata, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    maxCountour = max(contours, key=cv.contourArea)
    boundingRect = cv.minAreaRect(np.array(maxCountour))
    return boundingRect


def detectBoardAdaptiveGaussian(wdata):
    thresh = cv.adaptiveThreshold(wdata, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    maxCountour = max(contours, key=cv.contourArea)
    boundingRect = cv.minAreaRect(np.array(maxCountour))
    return boundingRect


def detectBoard(img, threshold_low):
    data = cv.imread(img)
    wdata = cv.imread(img, 0)

    boundingRect = detectBoardOtsu(wdata)
    boundingBox = np.int0(cv.boxPoints(boundingRect))

    cv.drawContours(data, [boundingBox], 0, (0, 0, 255), 5)
    cv.imshow("wbd", data)
    cv.waitKey()


# detectBoard("C:\\projects\\wbd\\dataset\\20201016_162039.jpg", 170)
# detectBoard("C:\\projects\\wbd\\dataset\\20201016_162105.jpg", 170)
# detectBoard("C:\\projects\\wbd\\dataset\\20201016_162136.jpg", 170)
# detectBoard("C:\\projects\\wbd\\dataset\\20201016_162201.jpg", 170)
# detectBoard("C:\\projects\\wbd\\dataset\\20201016_162258.jpg", 170)
detectBoard("C:\\projects\\wbd\\dataset\\20201016_163801.jpg", 170)
# detectBoard("C:\\projects\\wbd\\dataset\\20201016_163815.jpg", 170)
