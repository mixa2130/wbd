import json
from wbd import board_calibration, board_transform
import cv2 as cv
import numpy as np
import argparse
from ISR.models import RDN
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument('--image-path', help='Path to image file')
ap.add_argument('--image-url', help='Image URL')
ap.add_argument('--calibrate', help='Calibrate board points using GUI and save to specified file', default=False)
ap.add_argument('--transform', help='Transform board using points from specified file(s)', nargs='*')
ap.add_argument('--output', help='Save transformed boards to specified file(s)', nargs='*')
ap.add_argument('--superres', help='Apply super-resolution (slow!)', default=False, action='store_true')
ap.add_argument('--gui', help='Enable gui', default=False, action='store_true')
args = vars(ap.parse_args())

if args["calibrate"]:
    points = board_calibration.board_calibration(cv.imread(args["image"]))

    if len(points) == 4:
        with open(args["calibrate"], 'w') as f:
            json.dump(points, f)
    else:
        print("Expected 4 points, got " + str(len(points)))
elif args["transform"]:
    if args["superres"]:
        rdn = RDN(weights='psnr-small')

    if args["image_path"]:
        original = cv.imread(args["image_path"])
    elif args["image_url"]:
        res = urllib.request.urlopen(args["image_url"])
        original = np.asarray(bytearray(res.read()), dtype="uint8")
        original = cv.imdecode(original, cv.IMREAD_COLOR)
    else:
        print("Neither `--image-path` nor `--image-url` a specified, use `--help` to print usage")
        exit(1)

    if args["gui"]:
        cv.imshow("source", original)

    idx = 0
    for transform in args["transform"]:
        with open(transform) as f:
            points = np.array(json.load(f), dtype="float32")
            crop = board_transform.four_point_transform(original, points)

            if args["superres"]:
                result = rdn.predict(crop)
            else:
                result = crop

            if args["gui"]:
                cv.imshow(transform, result)

            if args["output"] and len(args["output"]) > idx:
                cv.imwrite(args["output"][idx], result)

        idx = idx + 1
    cv.waitKey(0)
else:
    print("Use `--help` to print usage")