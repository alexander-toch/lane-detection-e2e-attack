import cv2
import numpy as np


def get_steering_angle(keypoints, image_size: tuple) -> float:
    """_summary_

    Args:
        keypoints: the inference output in the format [[lane1_points],[lane2_points], ...], where lane1_points is a list of [x,y] coordinates (floats)
        image_size (tuple): 2-element tuple in the format (width, height)

    Returns:
        float: the steering angle in degrees
    """

    assert (
        len(keypoints) >= 2
    ), "We need at least 2 lanes to calculate the steering angle"

    # get the lanes to the left and right of the car
    left_lane, right_lane, _ = get_ego_lanes(keypoints, image_size)

    matrix, _ = get_transform_matrix(image_size)

    left_lane_bev = perspective_warp(left_lane, matrix)
    right_lane_bev = perspective_warp(right_lane, matrix)

    left_lane_poly = fit_lane(left_lane_bev)
    right_lane_poly = fit_lane(right_lane_bev)

    angle = calculate_radius(left_lane_poly, right_lane_poly, image_size)

    return angle


def get_ego_lanes(lanes, image_size: tuple):
    left_lane = np.array(lanes[0])
    right_lane = np.array(lanes[len(lanes) - 1])
    selected_indices = [0, len(lanes) - 1]
    center = image_size[0] / 2

    for i, l in enumerate(lanes):
        lane = np.array(l)

        avg = np.average(lane[:, 0])

        if avg < center and avg > np.average(left_lane[:, 0]):
            left_lane = lane
            selected_indices[0] = i
            continue

        if avg > center and avg < np.average(right_lane[:, 0]):
            right_lane = lane
            selected_indices[1] = i
            continue

    return left_lane, right_lane, selected_indices


def transform_keypoints(keypoints, image_size: tuple):
    return 0


def get_transform_matrix(image_size: tuple):
    # manually selected source and destination points
    src = np.float32(
        [
            [480, 500],
            [800, 500],
            [image_size[0] - 50, image_size[1]],
            [150, image_size[1]],
        ]
    )
    line_dst_offset = 300

    dst = np.float32(
        [
            [src[3][0] + line_dst_offset, 0],
            [src[2][0] - line_dst_offset, 0],
            [src[2][0] - line_dst_offset, src[2][1]],
            [src[3][0] + line_dst_offset, src[3][1]],
        ]
    )

    matrix = cv2.getPerspectiveTransform(src, dst)
    minv = cv2.getPerspectiveTransform(dst, src)
    return matrix, minv


def transformPoint(point, matrix):
    # https://stackoverflow.com/a/57400980
    px = (matrix[0][0] * point[0] + matrix[0][1] * point[1] + matrix[0][2]) / (
        (matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2])
    )
    py = (matrix[1][0] * point[0] + matrix[1][1] * point[1] + matrix[1][2]) / (
        (matrix[2][0] * point[0] + matrix[2][1] * point[1] + matrix[2][2])
    )
    return [px, py]


def perspective_warp(points, transform_matrix):
    result_points = []
    for point in points:
        point = transformPoint(point, transform_matrix)
        result_points.append(point)

    return np.array(result_points)


def fit_lane(lane):
    polynom = np.polyfit(lane[:, 0], lane[:, 1], 2)
    predict = np.poly1d(polynom)
    return predict


def calculate_radius(left_lane_poly, right_lane_poly, image_size: tuple):
    width = image_size[0]
    height = image_size[1]

    ym_per_pix = 0.000106  # meters per pixel in y dimension
    xm_per_pix = 0.000106  # meters per pixel in x dimension

    plot_x = np.linspace(0, width - 1, width)
    left_fit_y = left_lane_poly(plot_x)
    right_fit_y = right_lane_poly(plot_x)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(plot_x * xm_per_pix, left_fit_y * ym_per_pix, 2)
    right_fit_cr = np.polyfit(plot_x * xm_per_pix, right_fit_y * ym_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = (
        (1 + (2 * left_fit_cr[0] * height * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * left_fit_cr[0])

    right_curverad = (
        (1 + (2 * right_fit_cr[0] * height * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5
    ) / np.absolute(2 * right_fit_cr[0])

    r1 = round((float(left_curverad) + float(right_curverad)) / 2.0, 2)
    print(r1)
    if left_lane_poly.coefficients[0] - left_lane_poly.coefficients[-1] > 60:
        # curve_direction = 'Left'
        angle = -5729.57795 / r1
    elif left_lane_poly.coefficients[-1] - left_lane_poly.coefficients[0] > 60:
        # curve_direction = 'Right'
        angle = 5729.57795 / r1
    else:
        # curve_direction = 'Straight'
        angle = 5729.57795 / r1

    return angle
