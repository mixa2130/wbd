import numpy as np
import cv2 as cv


def detectBoardOtsu(wdata):
    _, thresh = cv.threshold(wdata, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv.contourArea)


def detectBoardAdaptiveGaussian(wdata):
    thresh = cv.adaptiveThreshold(wdata, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 0)
    contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    return max(contours, key=cv.contourArea)


def detectBoard(img_path):
    img_original = cv.imread(img_path)
    img = cv.imread(img_path, 0)

    contour = detectBoardOtsu(img)

    mask = np.zeros_like(img)  # Create mask where white is what we want, black otherwise
    cv.drawContours(mask, [contour], 0, 255, -1)  # Draw filled contour in mask
    out = np.zeros_like(img)  # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]

    # Now crop
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = img_original [topy:bottomy + 1, topx:bottomx + 1]

    # Show the output image
    cv.imshow('Output', out)
    cv.waitKey(0)
    cv.destroyAllWindows()


detectBoard("/home/kotborealis/projects/wbd/dataset/20201016_162039.jpg")
detectBoard("/home/kotborealis/projects/wbd/dataset/20201016_162105.jpg")
detectBoard("/home/kotborealis/projects/wbd/dataset/20201016_162136.jpg")
detectBoard("/home/kotborealis/projects/wbd/dataset/20201016_162201.jpg")
detectBoard("/home/kotborealis/projects/wbd/dataset/20201016_162258.jpg")
detectBoard("/home/kotborealis/projects/wbd/dataset/20201016_163815.jpg")
detectBoard("/home/kotborealis/projects/wbd/dataset/20201016_163801.jpg")
