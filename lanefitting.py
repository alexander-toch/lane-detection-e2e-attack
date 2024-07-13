import cv2
import numpy as np

def get_offset_center(keypoints, image_size: tuple, transform_matrix=None) -> float:
    """Calculates the offset from the center of the lane in meters

    Args:
        keypoints: the inference output in the format [[lane1_points],[lane2_points], ...], where lane1_points is a list of [x,y] coordinates (floats)
        image_size (tuple): 2-element tuple in the format (width, height)

    Returns:
        float: the offset from the center of the lane in meters
    """
    if len(keypoints) < 2:
        return None, None, None

    # get the lanes to the left and right of the car
    if transform_matrix is None:
        transform_matrix, _ = get_transform_matrix(image_size)
    left_lane_bev, right_lane_bev, _ = get_ego_lanes(keypoints, transform_matrix, image_size)

    left_lane_poly = fit_lane(left_lane_bev)
    right_lane_poly = fit_lane(right_lane_bev)

    width = image_size[0]
    height = image_size[1]

    left_lane_start = left_lane_poly(height)
    right_lane_start = right_lane_poly(height)
    current_center_x = left_lane_start + (right_lane_start - left_lane_start) / 2
    desired_center_x = width / 2.0 # center of the image (if camera is in the center of the car)

    # print(f"left_lane_start: {left_lane_start}, right_lane_start: {right_lane_start}, current_center_x: {current_center_x}, desired_center_x: {desired_center_x}")
    # print(left_lane_poly(last_10))
    # print(right_lane_poly(last_10))

    off_center = round(
        current_center_x - desired_center_x, 4
    )  # assume camera is in the center of the car

    # draw a virtual line from the center of the car to direction of the lane
    radius = calculate_radius(left_lane_poly, right_lane_poly, image_size)
    angle_rad = np.deg2rad(radius)

    start_point = (current_center_x, height)
    end_point = (
        desired_center_x,
        height - 400 * np.sin(angle_rad), # TODO: check if 400 needs to be dynamically calculated
    )
    heading_theta = np.arctan2(
        end_point[1] - start_point[1], end_point[0] - start_point[0]
    )

    debug_info = [radius, angle_rad, start_point, end_point]

    return off_center, heading_theta, debug_info

def get_ego_lanes(lanes, transformation_matrix, image_size: tuple):
    lanes_transformed = []
    for lane in lanes:       
       lanes_transformed.append(perspective_warp(np.array(lane), transformation_matrix))

    left_lane = np.float32(lanes_transformed[0])
    right_lane = np.float32(lanes_transformed[len(lanes_transformed)-1])
    selected_indices = [0, len(lanes_transformed)-1]
    middle = image_size[0]/2

    for i, lane in enumerate(lanes_transformed):
        avg = np.average(lane.T[0])
        # print(f"Lane {i}: Average: {avg}, Middle: {middle}, Left: {np.average(left_lane.T[0])}, Right: {np.average(right_lane.T[0])}")

        if avg < middle and avg > np.average(left_lane.T[0]) and avg < np.average(right_lane.T[0]):
            left_lane = np.float32(lane)
            selected_indices[0] = i
            continue

        if avg > middle and avg < np.average(right_lane.T[0]) and avg > np.average(left_lane.T[0]) and i != selected_indices[0]:
            right_lane = np.float32(lane)
            selected_indices[1] = i
            continue

    # TODO: find a better fix for the case when the lane detection is not accurate
    # print(f"Std right: {np.std(right_lane[:, 0])}, Std left: {np.std(left_lane[:, 0])}")
    # if np.std(right_lane[:, 0]) > 50:
    #     # find the point with the biggest x-axis difference to the previous point
    #     max_diff = 0
    #     max_diff_index = 0
    #     for i in range(1, len(right_lane)):
    #         diff = right_lane[i][0] - right_lane[i-1][0]
    #         if diff > max_diff:
    #             max_diff = diff
    #             max_diff_index = i
    #     new_r = right_lane[right_lane[:, 0] < right_lane[max_diff_index-1][0]-1]
    #     if len(new_r) > len(right_lane)/2:
    #         right_lane = new_r

    return left_lane, right_lane, selected_indices

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
        np.vstack((points[:,0], points[:,1])).T[np.newaxis, ...], transform_matrix
    )

    return transform[0]


def fit_lane(lane):
    import warnings
    warnings.simplefilter('ignore', np.RankWarning)
    polynom = np.polyfit(lane[:, 1], lane[:, 0], 2)  # note that we are fitting for y, x
    polynom_function = np.poly1d(
        polynom
    )  # we want to predict x values for a given height (y)
    return polynom_function


def calculate_radius(left_lane_poly, right_lane_poly, image_size: tuple):
    width = image_size[0]
    height = image_size[1]

    ym_per_pix = 0.00106  # meters per pixel in y dimension
    xm_per_pix = 0.00106  # meters per pixel in x dimension

    y_range = np.linspace(0, height - 1, height)
    left_fit = left_lane_poly(y_range)
    right_fit = right_lane_poly(y_range)

    left_fit_cr = np.polyfit(y_range * ym_per_pix, left_fit * xm_per_pix, 2)
    right_fit_cr = np.polyfit(y_range * ym_per_pix, right_fit * xm_per_pix, 2)

    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * height * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_cr[0])

    right_curverad = ((1 + (2 * right_fit_cr[0] * height * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_cr[0])

    radius1 = max(round((float(left_curverad) + float(right_curverad))/2.,2), 0.0001)

    if left_fit[0] - left_fit[-1] < -60:
        # curve_direction = 'Left'
        radius=-5729.57795/radius1
    elif left_fit[-1] - left_fit[0] > 60:
        # curve_direction = 'Right'
        radius=5729.57795/radius1
    else:
        # curve_direction = 'Straight'
        radius=5729.57795/radius1

    return radius


def draw_lane(image, keypoints, image_size: tuple, transform_matrix=None, draw_lane_overlay=True):
    if len(keypoints) < 2:
        print("No lanes detected")
        image_np = np.array(image)
        cv2.putText(image_np, f'NO LANES DETECTED', (10, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2, cv2.LINE_AA)
        return image_np

    if transform_matrix is None: 
        transform_matrix, _ = get_transform_matrix(image_size)

    left_lane_bev, right_lane_bev, selected_indices = get_ego_lanes(keypoints, transform_matrix, image_size)

    if (selected_indices[0] == selected_indices[1]):
        print("Warning: ego lane indexes are the same, this should not happen!")

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

    image_np = np.array(image) 

    colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,255,0), (255,0,255)]
    for i, lane in enumerate(keypoints):
        w = 3 if i in selected_indices else 1        
        for point in lane:
            cv2.circle(image_np, (int(point[0]), int(point[1])), w, colors[i], -1)

    offset_center, theta, debug_info = get_offset_center(keypoints, image_size, transform_matrix=transform_matrix)
    cv2.putText(image_np, f'Offset center: {offset_center}px (+ means deviation to right,- means to the left)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(image_np, f'Theta: {theta} ({np.rad2deg(theta)})', (10, 40), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    direction = "right" if offset_center < 0 else "left" if offset_center > 0 else "straight"
    cv2.putText(image_np, f'Steering direction: {direction}. Green circle is desired center. White is current lane center.', (10, 40), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # desired center
    cv2.circle(image_np, (int(width/2), int(height-3)), 3, (0, 255, 0), -1, cv2.LINE_AA)

    # current lane center
    # debug_info = {radius, angle_rad, start_point, end_point}
    left_lane_start = left_lane_poly(height)
    right_lane_start = right_lane_poly(height)
    current_center_x = left_lane_start + (right_lane_start - left_lane_start) / 2
    cv2.circle(image_np, (int(current_center_x), int(height-3)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    # end_point_transformed = perspective_warp(np.array([debug_info[3]]), transform_matrix_inv)[0]
    # cv2.arrowedLine(image_np, (int(width/2), int(height-3)), np.int_(end_point_transformed), (0, 0, 0), 1, cv2.LINE_AA)  

    if draw_lane_overlay:
        color_fill_image = cv2.fillPoly(color_fill_image, [pts], (0, 255, 0))  
        color_fill_image_transformed = cv2.warpPerspective(color_fill_image, transform_matrix, (width, height), flags=cv2.WARP_INVERSE_MAP)
        image_np = cv2.addWeighted(image_np, 1, color_fill_image_transformed, 0.2, 0, dtype=cv2.CV_8U)

    return image_np

def draw_lane_bev(image, keypoints, image_size: tuple, transform_matrix=None):
    if len(keypoints) < 2:
        print("No lanes detected")
        image_np = np.array(image)
        cv2.putText(image_np, f'NO LANES DETECTED', (10, 60), cv2.FONT_HERSHEY_SIMPLEX , 2, (0, 0, 255), 2, cv2.LINE_AA)
        return image_np

    if transform_matrix is None: 
        transform_matrix, _ = get_transform_matrix(image_size)

    left_lane_bev, right_lane_bev, selected_indices = get_ego_lanes(keypoints, transform_matrix, image_size)

    if (selected_indices[0] == selected_indices[1]):
        print("Warning: ego lane indexes are the same, this should not happen!")

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

    image_np = np.array(image)
    image_np = cv2.warpPerspective(image_np, transform_matrix, (width, height))

    colors = [(255,0,0), (0,255,0), (0,0,255), (0,255,255), (255,255,0), (255,0,255)]

    for i, lane in enumerate(keypoints):
        lane_bev = perspective_warp(np.array(lane), transform_matrix)
        w = 2 if i in selected_indices else 1        
        for point in lane_bev:
            cv2.circle(image_np, (int(point[0]), int(point[1])), w, colors[i], -1)

    offset_center, theta, debug_info = get_offset_center(keypoints, image_size, transform_matrix=transform_matrix)
    cv2.putText(image_np, f'Offset center: {offset_center}px (+ means deviation to right,- means to the left)', (10, 20), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    # cv2.putText(image_np, f'Theta: {theta} ({np.rad2deg(theta)})', (10, 40), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    direction = "left" if offset_center < 0 else "right" if offset_center > 0 else "straight"
    cv2.putText(image_np, f'Steering direction: {direction}. Green circle is desired center. White is current lane center.', (10, 40), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # desired center
    cv2.circle(image_np, (int(width/2), int(height-3)), 3, (0, 255, 0), -1, cv2.LINE_AA)

    # current lane center
    # debug_info = {radius, angle_rad, start_point, end_point}
    left_lane_start = left_lane_poly(height)
    right_lane_start = right_lane_poly(height)
    current_center_x = left_lane_start + (right_lane_start - left_lane_start) / 2
    cv2.circle(image_np, (int(current_center_x), int(height-3)), 3, (255, 255, 255), -1, cv2.LINE_AA)
    # end_point_transformed = perspective_warp(np.array([debug_info[3]]), transform_matrix_inv)[0]
    # cv2.arrowedLine(image_np, (int(width/2), int(height-3)), np.int_(end_point_transformed), (0, 0, 0), 1, cv2.LINE_AA)  

    color_fill_image = cv2.fillPoly(color_fill_image, [pts], (0, 255, 0))  
    result = cv2.addWeighted(image_np, 1, color_fill_image, 0.2, 0, dtype=cv2.CV_8U)
    return result

class Camera:
  K = np.zeros([3, 3])
  R = np.zeros([3, 3])
  t = np.zeros([3, 1])
  P = np.zeros([3, 4])

  def setK(self, fx, fy, px, py):
    self.K[0, 0] = fx
    self.K[1, 1] = fy
    self.K[0, 2] = px
    self.K[1, 2] = py
    self.K[2, 2] = 1.0

  def setR(self, y, p, r):

    Rz = np.array([[np.cos(-y), -np.sin(-y), 0.0], [np.sin(-y), np.cos(-y), 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[np.cos(-p), 0.0, np.sin(-p)], [0.0, 1.0, 0.0], [-np.sin(-p), 0.0, np.cos(-p)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(-r), -np.sin(-r)], [0.0, np.sin(-r), np.cos(-r)]])
    Rs = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]) # switch axes (x = -y, y = -z, z = x)
    self.R = Rs.dot(Rz.dot(Ry.dot(Rx)))

  def setT(self, XCam, YCam, ZCam):
    X = np.array([XCam, YCam, ZCam])
    self.t = -self.R.dot(X)

  def updateP(self):
    Rt = np.zeros([3, 4])
    Rt[0:3, 0:3] = self.R
    Rt[0:3, 3] = self.t
    self.P = self.K.dot(Rt)

  def __init__(self, config):
    self.config = config
    self.setK(config["fx"], config["fy"], config["px"], config["py"])
    self.setR(np.deg2rad(config["yaw"]), np.deg2rad(config["pitch"]), np.deg2rad(config["roll"]))
    self.setT(config["XCam"], config["YCam"], config["ZCam"])
    self.updateP()

def get_ipm_via_camera_config(image, fx, fy, res=1):
    """
    Get the Inverse Perspective Mapping (IPM) of an image using a camera configuration
    fx: focal length in x-direction [px]
    fy: focal length in y-direction [px]
    res [px/m]
    """
    width, height = image.shape[1], image.shape[0]

    config = {
        "fx": fx,	
        "fy": fy,
        "px": width / 2,
        "py": height / 2,
        'yaw': 90.0, 
        'pitch': 0.0, 
        'roll': 0.0, 
        'XCam': 0.0, 
        'YCam': 0.0, 
        'ZCam': 1.0
    }

    cam = Camera(config)

    cam_height = 50.0 # 50m high camera (drone)
    x_offset = 35.0 # in driving direction

    outputRes = (int(2 * cam.config["py"]), int(2 * cam.config["px"]))
    dx = outputRes[1] / cam.config["fx"] * cam_height
    dy = outputRes[0] / cam.config["fy"] * cam_height
    pxPerM = (outputRes[0] / dx, outputRes[1] / dy)


    # setup mapping from street/top-image plane to world coords
    shift = (outputRes[0] / 2.0, outputRes[1] / 2.0) # was (outputRes[0] / 2.0, outputRes[1] / 2.0)
    shift = shift[0] + x_offset * pxPerM[0], shift[1] - cam.config["XCam"] * pxPerM[1]
    M = np.array([[1.0 / pxPerM[1], 0.0, -shift[1] / pxPerM[1]], [0.0, -1.0 / pxPerM[0], shift[0] / pxPerM[0]], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    ipm = np.linalg.inv(cam.P.dot(M))
    return ipm