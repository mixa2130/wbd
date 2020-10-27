import json
from wbd import board_calibration, board_transform
import cv2 as cv
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--image', help='Path to image file')
ap.add_argument('--calibrate-board', help='Calibrate board points and save to specified file', default=False)
ap.add_argument('--transform-board', help='Transform board using points from specified file', nargs='*')
args = vars(ap.parse_args())

if args["calibrate_board"]:
    points = board_calibration.board_calibration(cv.imread(args["image"]))

    if len(points) == 4:
        with open(args["calibrate_board"], 'w') as f:
            json.dump(points, f)
    else:
        print("Expected 4 points, got " + str(len(points)))
elif args["transform_board"]:
    original = cv.imread(args["image"])
    cv.imshow(args["image"], original)

    for transform in args["transform_board"]:
        print(transform)
        with open(transform) as f:
            points = np.array(json.load(f), dtype="float32")
            result = board_transform.four_point_transform(original, points)
            cv.imshow(transform, result)
    cv.waitKey(0)