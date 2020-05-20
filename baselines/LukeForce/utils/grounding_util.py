
def get_bbox_from_keypoints(projected_kp, img_w, img_h, scale_ratio=None):
    """
    Get a bounding box from projected keypoints.
    :param projected_kp: K * 2 list. Kps are in [x, y] format.
    :param img_w: image width.
    :param img_h: image height.
    :return: bounding box in format of [x1, y1, x2, y2].
    """
    all_x, all_y = [], []
    for idx, one_kp in enumerate(projected_kp):
        x, y = one_kp
        all_x.append(x)
        all_y.append(y)
    x_min, x_max, y_min, y_max = min(all_x), max(all_x), min(all_y), max(all_y)
    center_x = int((x_min + x_max) / 2)
    center_y = int((y_min + y_max) / 2)
    width = 2 * (x_max - center_x)
    height = 2 * (y_max - center_y)
    if scale_ratio is not None:
        scaled_width = scale_ratio * width
        scaled_height = scale_ratio * height
        final_xmin = max(0, center_x - scaled_width / 2)
        final_xmax = min(img_w, center_x + scaled_width / 2)
        final_ymin = max(0, center_y - scaled_height / 2)
        final_ymax = min(img_h, center_y + scaled_height / 2)
        return [int(final_xmin), int(final_ymin), int(final_xmax), int(final_ymax)]
    else:
        return [int(x_min), int(y_min), int(x_max), int(y_max)]
