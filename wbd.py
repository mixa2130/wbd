import json
from wbd import board_calibration, board_transform
import cv2 as cv
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--image', help='Path to image file')
ap.add_argument('--calibrate-board', help='Calibrate board points and save to specified file', default=False)
ap.add_argument('--transform-board', help='Transform board using points from specified file', default=False)
args = vars(ap.parse_args())

if args["calibrate_board"]:
    points = board_calibration.board_calibration(cv.imread(args["image"]))

    if len(points) == 4:
        with open(args["calibrate_board"], 'w') as f:
            json.dump(points, f)
    else:
        print("Expected 4 points, got " + str(len(points)))
elif args["transform_board"]:
    points = []
    with open(args["transform_board"]) as f:
        points = np.array(json.load(f), dtype="float32")

    original = cv.imread(args["image"])
    result = board_transform.four_point_transform(original, points)
    cv.imshow("Original", original)
    cv.waitKey(0)
    cv.imshow("Warped", result)
    cv.waitKey(0)