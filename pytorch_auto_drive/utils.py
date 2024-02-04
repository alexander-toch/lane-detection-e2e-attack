import torch
import numpy as np
import cv2


@torch.no_grad()
def lane_as_segmentation_inference(
    net, inputs, input_sizes, gap, ppl, thresh, dataset, max_lane=0, forward=True
):
    # Assume net and images are on the same device
    # images: B x C x H x W
    # Return: a list of lane predictions on each image
    outputs = (
        net(inputs) if forward else inputs
    )  # Support no forwarding inside this function
    prob_map = torch.nn.functional.interpolate(
        outputs["out"], size=input_sizes[0], mode="bilinear", align_corners=True
    ).softmax(dim=1)
    existence_conf = outputs["lane"].sigmoid()
    existence = existence_conf > 0.5
    if max_lane != 0:  # Lane max number prior for testing
        existence, existence_conf = lane_pruning(
            existence, existence_conf, max_lane=max_lane
        )

    prob_map = prob_map.cpu().numpy()
    existence = existence.cpu().numpy()

    # Get coordinates for lanes
    lane_coordinates = []
    for j in range(existence.shape[0]):
        lane_coordinates.append(
            prob_to_lines(
                prob_map[j],
                existence[j],
                resize_shape=input_sizes[1],
                gap=gap,
                ppl=ppl,
                thresh=thresh,
                dataset=dataset,
            )
        )

    return lane_coordinates


def prob_to_lines(
    seg_pred,
    exist,
    resize_shape=None,
    smooth=True,
    gap=20,
    ppl=None,
    thresh=0.3,
    dataset="culane",
):
    """
    Arguments:
    ----------
    seg_pred: np.array size (num_classes, h, w)
    resize_shape:  reshape size target, (H, W)
    exist:   list of existence, e.g. [0, 1, 1, 0]
    smooth:  whether to smooth the probability or not
    gap: y pixel gap for sampling
    ppl:     how many points for one lane
    thresh:  probability threshold
    all_points: Whether to save all sample points or just points predicted as lane
    Return:
    ----------
    coordinates: [x, y] list of lanes, e.g.: [ [[9, 569], [50, 549]] ,[[630, 569], [647, 549]] ]
    """
    if resize_shape is None:
        resize_shape = seg_pred.shape[1:]  # seg_pred (num_classes, h, w)
    _, h, w = seg_pred.shape
    H, W = resize_shape
    coordinates = []

    if ppl is None:
        ppl = round(H / 2 / gap)

    for i in range(1, seg_pred.shape[0]):
        prob_map = seg_pred[i, :, :]
        if exist[i - 1]:
            if smooth:
                prob_map = cv2.blur(prob_map, (9, 9), borderType=cv2.BORDER_REPLICATE)
            coords = get_lane(prob_map, gap, ppl, thresh, resize_shape, dataset=dataset)
            if coords.sum() == 0:
                continue
            if (
                dataset == "tusimple"
            ):  # Invalid sample points need to be included as negative value, e.g. -2
                coordinates.append(
                    [
                        (
                            [coords[j], H - (ppl - j) * gap]
                            if coords[j] > 0
                            else [-2, H - (ppl - j) * gap]
                        )
                        for j in range(ppl)
                    ]
                )
            elif dataset in ["culane", "llamas"]:
                coordinates.append(
                    [[coords[j], H - j * gap - 1] for j in range(ppl) if coords[j] > 0]
                )
            else:
                raise ValueError

    return coordinates


# Adapted from harryhan618/SCNN_Pytorch
# Note that in tensors we have indices start from 0 and in annotations coordinates start at 1
def get_lane(prob_map, gap, ppl, thresh, resize_shape=None, dataset="culane"):
    """
    Arguments:
    ----------
    prob_map: prob map for single lane, np array size (h, w)
    resize_shape:  reshape size target, (H, W)
    Return:
    ----------
    coords: x coords bottom up every gap px, 0 for non-exist, in resized shape
    """

    if resize_shape is None:
        resize_shape = prob_map.shape
    h, w = prob_map.shape
    H, W = resize_shape
    coords = np.zeros(ppl)
    for i in range(ppl):
        if dataset == "tusimple":  # Annotation start at 10 pixel away from bottom
            y = int(h - (ppl - i) * gap / H * h)
        elif dataset in ["culane", "llamas"]:  # Annotation start at bottom
            y = int(h - i * gap / H * h - 1)  # Same as original SCNN code
        else:
            raise ValueError
        if y < 0:
            break
        line = prob_map[y, :]
        id = np.argmax(line)
        if line[id] > thresh:
            coords[i] = int(id / w * W)
    if (coords > 0).sum() < 2:
        coords = np.zeros(ppl)
    return coords


def lane_pruning(existence, existence_conf, max_lane):
    # Prune lanes based on confidence (a max number constrain for lanes in an image)
    # Maybe too slow (but should be faster than topk/sort),
    # consider batch size >> max number of lanes
    while (existence.sum(dim=1) > max_lane).sum() > 0:
        indices = (existence.sum(dim=1, keepdim=True) > max_lane).expand_as(
            existence
        ) * (existence_conf == existence_conf.min(dim=1, keepdim=True).values)
        existence[indices] = 0
        existence_conf[indices] = 1.1  # So we can keep using min

    return existence, existence_conf


def lane_detection_visualize_batched(
    images,
    masks=None,
    keypoints=None,
    mask_colors=None,
    keypoint_color=None,
    std=None,
    mean=None,
    control_points=None,
    gt_keypoints=None,
    style="point",
    line_trans=0.4,
    compare_gt_metric="culane",
):
    # Draw images + lanes from tensors (batched)
    # None masks/keypoints and keypoints (x < 0 or y < 0) will be ignored
    # images (4D), masks (3D), keypoints (4D), colors (2D), std, mean: torch.Tensor
    # keypoints can be either List[List[N x 2 numpy array]] (for variate length lanes) or a 4D numpy array
    # filenames: List[str]
    # keypoint_color: RGB
    BGR_RED = [0, 0, 255]
    BGR_GREEN = [0, 255, 0]
    BGR_BLUE = [255, 0, 0]

    if keypoints is not None:
        if masks is None:
            images = images.permute(0, 2, 3, 1)
        if std is not None and mean is not None:
            images = images.float() * std + mean
        images = images.clamp_(0.0, 1.0) * 255.0
        images = images[..., [2, 1, 0]].cpu().numpy().astype(np.uint8)
        if keypoint_color is None:
            keypoint_color = [0, 0, 0]  # Black (sits well with lane colors)
        else:
            keypoint_color = keypoint_color[::-1]  # To BGR

        # Draw
        for i in range(images.shape[0]):

            if style == "point":
                if gt_keypoints is not None:
                    images[i] = draw_points(images[i], gt_keypoints[i], BGR_BLUE)
                images[i] = draw_points(images[i], keypoints[i], keypoint_color)
            elif style in ["line", "bezier"]:
                overlay = images[i].copy()
                if gt_keypoints is not None:
                    overlay = draw_points_as_lines(overlay, gt_keypoints[i], BGR_BLUE)
                overlay = draw_points_as_lines(overlay, keypoints[i], keypoint_color)
                images[i] = (
                    images[i].astype(float) * line_trans
                    + overlay.astype(float) * (1 - line_trans)
                ).astype(np.uint8)
                if style == "bezier":
                    assert (
                        control_points is not None
                    ), "Must provide control points for style bezier!"
                    images[i] = draw_points(
                        images[i], control_points[i], keypoint_color
                    )
            else:
                raise ValueError(
                    "Unknown keypoint visualization style: {}\nPlease use point/line/bezier".format(
                        style
                    )
                )
        images = images[..., [2, 1, 0]]

    return images


def draw_points(image, points, colors, radius=5, thickness=-1):
    # Draw lines (defined by points) on an image as keypoints
    # colors: can be a list that defines different colors for each line
    for j in range(len(points)):
        temp = points[j][(points[j][:, 0] > 0) * (points[j][:, 1] > 0)]
        for k in range(temp.shape[0]):
            color = colors[j] if isinstance(colors[0], list) else colors
            cv2.circle(
                image,
                (int(temp[k][0]), int(temp[k][1])),
                radius=radius,
                color=color,
                thickness=thickness,
            )
    return image


def draw_points_as_lines(image, points, colors, thickness=3):
    # Draw lines (defined by points) on an image by connecting points to lines
    # colors: can be a list that defines different colors for each line
    for j in range(len(points)):
        temp = points[j][(points[j][:, 0] > 0) * (points[j][:, 1] > 0)]
        for k in range(temp.shape[0] - 1):
            color = colors[j] if isinstance(colors[0], list) else colors
            cv2.line(
                image,
                (int(temp[k][0]), int(temp[k][1])),
                (int(temp[k + 1][0]), int(temp[k + 1][1])),
                color=color,
                thickness=thickness,
            )
    return image
