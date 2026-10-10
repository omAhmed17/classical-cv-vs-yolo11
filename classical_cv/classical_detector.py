import cv2 as cv
import numpy as np


def create_color_masks(image):
    """
    Convert the input image to HSV and create
    separate masks for Red, Blue and Green.
    """

    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # -----------------------------
    # Blue
    # -----------------------------
    lower_blue = np.array([100, 150, 20])
    upper_blue = np.array([140, 255, 255])

    mask_blue = cv.inRange(
        image_hsv,
        lower_blue,
        upper_blue
    )


    lower_red1 = np.array([0, 100, 20])
    upper_red1 = np.array([10, 255, 255])

    lower_red2 = np.array([170, 100, 20])
    upper_red2 = np.array([179, 255, 255])

    mask_red1 = cv.inRange(
        image_hsv,
        lower_red1,
        upper_red1
    )

    mask_red2 = cv.inRange(
        image_hsv,
        lower_red2,
        upper_red2
    )

    mask_red = cv.bitwise_or(
        mask_red1,
        mask_red2
    )

    # -----------------------------
    # Green
    # -----------------------------
    lower_green = np.array([35, 100, 20])
    upper_green = np.array([85, 255, 255])

    mask_green = cv.inRange(
        image_hsv,
        lower_green,
        upper_green
    )

    return {
        "Red": mask_red,
        "Blue": mask_blue,
        "Green": mask_green
    }


def classify_shape(contour, epsilon_ratio=0.04):
    """
    Classify a contour using polygon approximation.
    """

    perimeter = cv.arcLength(contour, True)

    epsilon = epsilon_ratio * perimeter

    approx = cv.approxPolyDP(
        contour,
        epsilon,
        True
    )

    corners = len(approx)

    if corners == 3:
        shape_name = "Triangle"

    elif corners == 4:
        shape_name = "Square"

    else:
        shape_name = "Circle"

    return shape_name, approx


def detect_objects(
    image,
    area_threshold=300,
    epsilon_ratio=0.04
):
    """
    Classical CV detector.

    Returns a list containing:
    - color
    - shape
    - class name
    - centroid
    - area
    - bounding box
    """

    color_masks = create_color_masks(image)

    detections = []

    for color_name, mask in color_masks.items():

        contours, _ = cv.findContours(
            mask,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE
        )

        for contour in contours:

            # -------------------------
            # Area filtering
            # -------------------------
            area = cv.contourArea(contour)

            if area < area_threshold:
                continue

            # -------------------------
            # Shape classification
            # -------------------------
            shape_name, approx = classify_shape(
                contour,
                epsilon_ratio
            )

            # -------------------------
            # Centroid
            # -------------------------
            M = cv.moments(contour)

            if M["m00"] == 0:
                continue

            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # -------------------------
            # Bounding box
            # -------------------------
            x, y, w, h = cv.boundingRect(contour)

            x1 = x
            y1 = y
            x2 = x + w
            y2 = y + h

            class_name = (
                f"{color_name.lower()}_"
                f"{shape_name.lower()}"
            )

            detections.append({
                "color": color_name.lower(),
                "shape": shape_name.lower(),
                "class_name": class_name,
                "center": (cx, cy),
                "area": area,
                "bbox": (x1, y1, x2, y2),
                "approx": approx
            })

    return detections


def draw_detections(image, detections):
    """
    Draw predictions on a copy of the image.
    """

    result = image.copy()

    for detection in detections:

        approx = detection["approx"]

        cx, cy = detection["center"]

        label = (
            f"{detection['color']} "
            f"{detection['shape']}"
        )

        cv.drawContours(
            result,
            [approx],
            -1,
            (0, 255, 0),
            3
        )

        cv.putText(
            result,
            label,
            (cx - 40, cy - 10),
            cv.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            2
        )

    return result