"""
https://learnopencv.com/camera-calibration-using-opencv/
https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html
https://medium.com/vacatronics/3-ways-to-calibrate-your-camera-using-opencv-and-python-395528a51615

CalibrationCoefficients:
camera_matrix - Внутренняя матрица камеры.
dist_coeffs - Коэффициенты искажения объектива.
rotation_vector - Вращение указано как вектор 3×13 × 13×1.
        Направление вектора задает ось вращения, а величина вектора — угол поворота
translation_vector - 3×1 Translation vector(вектор смещения).
"""
import glob
from typing import NamedTuple
from pathlib import Path

import numpy as np
import cv2

CHECKERBOARD = (6, 9)
CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

current_dir = str(Path.cwd())
home_prj_dir_path = current_dir[:current_dir.find('wbd') + 3]


class CalibrationCoefficients(NamedTuple):
    """Camera calibration coefficients"""
    ret: float
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rotation_vector: list
    translation_vector: list


def _draw_lines_on_chessboard(img, corners, filename):
    """Using the obtained corners draws lines on a chessboard"""
    img = cv2.drawChessboardCorners(img, CHECKERBOARD, corners, True)
    cv2.imwrite(f'{home_prj_dir_path}/calibration/lines_on_chessboard/{filename}', img=img)


def _camera_calibration(debug_mode: bool = False) -> CalibrationCoefficients:
    """Gets calibration coefficients using test images(from ./calibration directory) with chessboard pattern."""
    if debug_mode:
        Path(f"{home_prj_dir_path}/calibration/lines_on_chessboard").mkdir(parents=True, exist_ok=True)

    # Вектор для хранения векторов трехмерных точек для каждого изображения шахматной доски
    objpoints = []
    # Вектор для хранения векторов 2D точек для каждого изображения шахматной доски
    imgpoints = []

    # Определение мировых координат для 3D точек
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    images = glob.glob(f'{home_prj_dir_path}/calibration/chessboard_calibration_imgs/*.jpg')
    for file in images:
        img = cv2.imread(file)
        img_in_gray_colors = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Поиск углов шахматной доски
        pattern_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
        pattern_found, corners = cv2.findChessboardCorners(img_in_gray_colors, CHECKERBOARD, pattern_flags)

        if pattern_found:
            objpoints.append(objp)
            # уточнение координат пикселей для заданных 2d точек.
            corners2 = cv2.cornerSubPix(img_in_gray_colors, corners, (11, 11), (-1, -1), CRITERIA)

            imgpoints.append(corners2)
            if debug_mode:
                _draw_lines_on_chessboard(img=img, corners=corners2, filename=file[file.rfind('/') + 1:])

        # Разрешение изображения
        if prev_img_shape is None:
            prev_img_shape = img_in_gray_colors.shape[::-1]

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, prev_img_shape,
                                                       None, None)

    return CalibrationCoefficients(ret, mtx, dist, rvecs, tvecs)


def _save_coefficients(coefficients: CalibrationCoefficients,
                       weights_filename=f'{home_prj_dir_path}/weights'):
    """Save the camera matrix and the distortion coefficients to given path/file."""
    cv_file = cv2.FileStorage(weights_filename, cv2.FILE_STORAGE_WRITE)
    cv_file.write('K', coefficients.camera_matrix)
    cv_file.write('D', coefficients.dist_coeffs)

    # note you *release* you don't close() a FileStorage object
    cv_file.release()


def load_coefficients(weights_file_path=f'{home_prj_dir_path}/weights'):
    """Loads camera matrix and distortion coefficients."""
    cv_file = cv2.FileStorage(weights_file_path, cv2.FILE_STORAGE_READ)

    # note we also have to specify the type to retrieve other wise we only get a
    # FileNode object back instead of a matrix
    camera_matrix = cv_file.getNode('K').mat()
    dist_matrix = cv_file.getNode('D').mat()

    cv_file.release()
    return camera_matrix, dist_matrix


def undistort_img(filename: str, output_path: str):
    """
    Undistort image using weights from yaml file

    :param filename: path to the source file
    :param output_path: where to save the file after processing(transmitted without output filename, only path)
    """
    mtx, dist = load_coefficients()

    original = cv2.imread(filename)
    dst = cv2.undistort(original, mtx, dist, None, None)

    output_file_path: str = output_path + filename[filename.rfind('/') + 1:]
    cv2.imwrite(output_file_path, dst)
