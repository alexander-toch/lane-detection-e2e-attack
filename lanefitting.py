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

    if len(keypoints) < 2:
        return None

    # get the lanes to the left and right of the car
    left_lane, right_lane, _ = get_ego_lanes(keypoints, image_size)

    matrix, _ = get_transform_matrix(image_size)

    left_lane_bev = perspective_warp(left_lane, matrix)
    right_lane_bev = perspective_warp(right_lane, matrix)

    left_lane_poly = fit_lane(left_lane_bev)
    right_lane_poly = fit_lane(right_lane_bev)

    angle = calculate_radius(left_lane_poly, right_lane_poly, image_size)

    return angle


def get_offset_center(keypoints, image_size: tuple) -> float:
    """Calculates the offset from the center of the lane in meters

    Args:
        keypoints: the inference output in the format [[lane1_points],[lane2_points], ...], where lane1_points is a list of [x,y] coordinates (floats)
        image_size (tuple): 2-element tuple in the format (width, height)

    Returns:
        float: the offset from the center of the lane in meters
    """
    if len(keypoints) < 2:
        return None

    # get the lanes to the left and right of the car
    left_lane, right_lane, _ = get_ego_lanes(keypoints, image_size)

    matrix, matrix_inv = get_transform_matrix(image_size)

    left_lane_bev = perspective_warp(left_lane, matrix)
    right_lane_bev = perspective_warp(right_lane, matrix)

    left_lane_poly = fit_lane(left_lane_bev)
    right_lane_poly = fit_lane(right_lane_bev)

    width = image_size[0]
    height = image_size[1]

    left_lane_start = left_lane_poly(height)
    right_lane_start = right_lane_poly(height)
    current_center_x = left_lane_start + (right_lane_start - left_lane_start) / 2
    desired_center_x = width / 2.0

    xm_per_pix = 0.000106  # meters per pixel in x dimension
    off_center = round(
        (current_center_x - desired_center_x) * xm_per_pix, 4
    )  # assume camera is in the center of the car

    # draw a virtual line from the center of the car to direction of the lane
    angle_rad = np.deg2rad(
        radius - 90
    )  # TODO: check if this is correct on a right turn
    start_point = (desired_center_x, height)
    end_point = (
        current_center_x + 400 * np.cos(angle_rad),
        height + 400 * np.sin(angle_rad),
    )
    heading_theta = np.arctan2(
        end_point[1] - start_point[1], end_point[0] - start_point[0]
    )

    return off_center, heading_theta


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
            [470, 500],
            [800, 500],
            [image_size[0] - 50, image_size[1]],
            [150, image_size[1]],
        ]
    )
    line_dst_offset = 200

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


def perspective_warp(points, transform_matrix):
    transform = cv2.perspectiveTransform(
        np.vstack((points[0], points[1])).T[np.newaxis, ...], transform_matrix
    )
    transform = [transform[0][:, 0], transform[0][:, 1]]

    return transform


def fit_lane(lane):
    polynom = np.polyfit(lane[:, 1], lane[:, 0], 2)  # note that we are fitting for y, x
    polynom_function = np.poly1d(
        polynom
    )  # we want to predict x values for a given height (y)
    return polynom_function


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


def draw_lane(image, keypoints, image_size: tuple):
    if len(keypoints) < 2:
        return None

    left_lane, right_lane, _ = get_ego_lanes(keypoints, image_size)

    matrix, matrix_inv = get_transform_matrix(image_size)

    left_lane_bev = perspective_warp(left_lane, matrix)
    right_lane_bev = perspective_warp(right_lane, matrix)

    left_lane_poly = fit_lane(left_lane_bev)
    right_lane_poly = fit_lane(right_lane_bev)

    width = image_size[0]
    height = image_size[1]
    color_fill_image = np.zeros([height, width, 3])

    y_range = np.linspace(0, height - 1, height)
    left_fit = left_lane_poly(y_range)
    right_fit = right_lane_poly(y_range)

    l1 = np.transpose(np.vstack([left_fit, y_range]))
    l2 = np.flip(np.transpose(np.vstack([right_fit, y_range])), axis=0)
    pts = np.int_(np.vstack((l1, l2)))

    color_fill_image = cv2.fillPoly(color_fill_image, [pts], (0, 255, 0))

    image_np = np.array(image)

    color_fill_image_transformed = cv2.warpPerspective(
        color_fill_image, matrix_inv, (width, height)
    )
    result = cv2.addWeighted(
        image_np, 1, color_fill_image_transformed, 0.2, 0, dtype=cv2.CV_8U
    )

    return result
