from .calibration import _save_coefficients, _camera_calibration
from .calibration import undistort_img


def get_weights_for_camera_calibration(debug_mode: bool):
    calibration_coefficients = _camera_calibration(debug_mode=debug_mode)
    _save_coefficients(calibration_coefficients)  # Will create a yaml 'weights' file with calibration coefficients
