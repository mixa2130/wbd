import json
from wbd import board_calibration, board_transform
import cv2 as cv
import numpy as np
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('--image', help='Path to image file')
ap.add_argument('--calibrate-board', help='Calibrate board points and save to specified file', default=False)
ap.add_argument('--transform-board', help='Transform board using points from specified file(s)', nargs='*')
ap.add_argument('--save-as', help='Save transformed boards to specified file(s)', nargs='*')
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

    idx = 0
    for transform in args["transform_board"]:
        with open(transform) as f:
            points = np.array(json.load(f), dtype="float32")
            result = board_transform.four_point_transform(original, points)
            if args["save_as"] and len(args["save_as"]) > idx:
                cv.imwrite(args["save_as"][idx], result)

            cv.imshow(transform, result)
        idx = idx + 1
    cv.waitKey(0)