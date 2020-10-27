import cv2 as cv


def board_calibration(img):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    points = []

    def m_callback(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            cv.circle(img, (x, y), 10, colors[len(points)], -1)
            points.append((x, y))

    cv.namedWindow('Calibrate board')
    cv.setMouseCallback('Calibrate board', m_callback)

    while 1:
        cv.imshow('Calibrate board', img)
        if cv.waitKey(1) & 0xFF == 27 or len(points) == 4:
            break
    cv.destroyAllWindows()

    if len(points) == 4:
        return points
