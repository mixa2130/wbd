import json
from wbd import board_calibration, board_transform
import cv2 as cv
import numpy as np
import argparse
import requests

ap = argparse.ArgumentParser()
ap.add_argument('--image-path', help='Grab image from specified file')
ap.add_argument('--image-url', help='Grab image from specified URL')
ap.add_argument('--image-rtsp', help='Grab image from specified RTSP stream')
ap.add_argument('--calibrate', help='Calibrate board points using GUI and save to specified file', default=False)
ap.add_argument('--transform', help='Transform board using points from specified file(s)', nargs='*')
ap.add_argument('--output', help='Save transformed boards to specified file(s)', nargs='*')
ap.add_argument('--output-original', help='Save original to specified file(s)')
args = vars(ap.parse_args())

original = []

if args["image_path"]:
    original = cv.imread(args["image_path"])
elif args["image_url"]:
    res = requests.get(args["image_url"], stream=True).content
    original = np.asarray(bytearray(res), dtype="uint8")
    original = cv.imdecode(original, cv.IMREAD_COLOR)
elif args["image_rtsp"]:
    res, frame = cv.VideoCapture(args["image_rtsp"]).read()
    if not res:
        print("Failed to grab frame from specified rtsp stream")
        exit(1)
else:
    print("Neither `--image-path`, `--image-url` or `--image-rtsp` was specified, use `--help` to print usage")
    exit(1)

if args["calibrate"]:
    points = board_calibration.board_calibration(original)

    if len(points) == 4:
        with open(args["calibrate"], 'w') as f:
            json.dump(points, f)
    else:
        print("Expected 4 points, got " + str(len(points)))
elif args["transform"]:
    idx = 0
    for transform in args["transform"]:
        with open(transform) as f:
            points = np.array(json.load(f), dtype="float32")
            result = board_transform.four_point_transform(original, points)

            if args["output"] and len(args["output"]) > 0:
                cv.imwrite(args["output"].pop(0), result)

        idx = idx + 1

if args["output_original"]:
    if args["image_rtsp"]:
        res, frame = cv.VideoCapture(args["image_rtsp"]).read()
        cv.imwrite(args["output_original"], frame)
