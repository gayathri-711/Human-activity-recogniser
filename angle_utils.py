import math
import numpy as np


def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle ABC formed by three 2D points.

    Parameters
    ----------
    point_a : array-like
        First point (x, y).

    point_b : array-like
        Vertex point (x, y).

    point_c : array-like
        Third point (x, y).

    Returns
    -------
    float
        Angle in degrees between 0 and 180.
    """

    try:
        a = np.asarray(point_a, dtype=np.float32)
        b = np.asarray(point_b, dtype=np.float32)
        c = np.asarray(point_c, dtype=np.float32)

        # Vectors BA and BC
        ba = a - b
        bc = c - b

        # Vector magnitudes
        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)

        # Prevent division by zero
        if magnitude_ba < 1e-6 or magnitude_bc < 1e-6:
            return 0.0

        # Dot-product formula
        cosine_value = np.dot(ba, bc) / (
            magnitude_ba * magnitude_bc
        )

        # Numerical safety
        cosine_value = np.clip(
            cosine_value,
            -1.0,
            1.0
        )

        # Convert radians → degrees
        angle = math.degrees(
            math.acos(cosine_value)
        )

        return float(angle)

    except Exception as error:
        print(
            f"Angle calculation error: {error}"
        )

        return 0.0


def calculate_distance(point_a, point_b):
    """
    Calculate Euclidean distance between two points.
    """

    try:
        a = np.asarray(point_a, dtype=np.float32)
        b = np.asarray(point_b, dtype=np.float32)

        return float(
            np.linalg.norm(a - b)
        )

    except Exception:
        return 0.0


def calculate_midpoint(point_a, point_b):
    """
    Calculate the midpoint between two 2D points.
    """

    try:
        return (
            (
                float(point_a[0]) +
                float(point_b[0])
            ) / 2.0,

            (
                float(point_a[1]) +
                float(point_b[1])
            ) / 2.0
        )

    except Exception:
        return (0.0, 0.0)


def calculate_vertical_angle(point_a, point_b):
    """
    Calculate the orientation of the line A-B
    relative to the vertical axis.

    Returns an angle in degrees.

    0 degrees   = vertical
    90 degrees  = horizontal
    """

    try:
        dx = float(point_b[0]) - float(point_a[0])
        dy = float(point_b[1]) - float(point_a[1])

        angle = math.degrees(
            math.atan2(
                abs(dx),
                abs(dy) + 1e-6
            )
        )

        return float(angle)

    except Exception:
        return 90.0


def calculate_horizontal_angle(point_a, point_b):
    """
    Calculate the orientation of the line A-B
    relative to the horizontal axis.

    Returns an angle in degrees.

    0 degrees   = horizontal
    90 degrees  = vertical
    """

    try:
        dx = float(point_b[0]) - float(point_a[0])
        dy = float(point_b[1]) - float(point_a[1])

        angle = math.degrees(
            math.atan2(
                abs(dy),
                abs(dx) + 1e-6
            )
        )

        return float(angle)

    except Exception:
        return 90.0