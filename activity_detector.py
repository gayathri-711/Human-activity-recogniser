from collections import deque
import numpy as np

from angle_utils import (
    calculate_angle,
    calculate_midpoint,
    calculate_vertical_angle,
    calculate_horizontal_angle
)


class ActivityDetector:
    """
    Human Activity Recognition engine.

    Detects:
        - STANDING
        - SITTING
        - LYING DOWN
        - WALKING
        - RUNNING
        - NO PERSON
        - BODY NOT FULLY VISIBLE

    The detector uses:
        1. YOLO pose landmarks
        2. Landmark confidence
        3. Body geometry
        4. Joint angles
        5. Movement history
        6. Temporal smoothing
    """

    NO_PERSON = "NO PERSON"

    BODY_NOT_VISIBLE = "BODY NOT FULLY VISIBLE"

    STANDING = "STANDING"

    SITTING = "SITTING"

    LYING = "LYING DOWN"

    WALKING = "WALKING"

    RUNNING = "RUNNING"

    def __init__(self):

        # Current activity
        self.current_activity = self.NO_PERSON

        # Recent activity predictions
        self.activity_history = deque(
            maxlen=10
        )

        # Previous body positions
        self.previous_hip = None

        self.previous_shoulder = None

        self.previous_ankle = None

        # Movement history
        self.movement_history = deque(
            maxlen=15
        )

        # Confidence
        self.confidence = 0.0

        # Number of frames processed
        self.frame_count = 0

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):

        self.current_activity = (
            self.NO_PERSON
        )

        self.activity_history.clear()

        self.movement_history.clear()

        self.previous_hip = None

        self.previous_shoulder = None

        self.previous_ankle = None

        self.confidence = 0.0

        self.frame_count = 0

    # ======================================================
    # LANDMARK VALIDATION
    # ======================================================

    def is_landmark_valid(
        self,
        confidence,
        index,
        threshold=0.45
    ):
        """
        Check whether a YOLO landmark has
        sufficient confidence.
        """

        try:

            if confidence is None:
                return False

            if index >= len(confidence):
                return False

            value = float(
                confidence[index]
            )

            return value >= threshold

        except Exception:

            return False

    # ======================================================
    # BODY VISIBILITY
    # ======================================================

    def check_body_visibility(
        self,
        confidence
    ):
        """
        Check whether enough important
        body landmarks are visible.

        Required:
            shoulders
            hips
            knees
            ankles
        """

        required_landmarks = [
            5,   # Left Shoulder
            6,   # Right Shoulder

            11,  # Left Hip
            12,  # Right Hip

            13,  # Left Knee
            14,  # Right Knee

            15,  # Left Ankle
            16   # Right Ankle
        ]

        visible = 0

        for index in required_landmarks:

            if self.is_landmark_valid(
                confidence,
                index
            ):

                visible += 1

        # At least 6 of the 8 important
        # landmarks must be visible.
        return visible >= 6

    # ======================================================
    # UPPER BODY VISIBILITY
    # ======================================================

    def check_upper_body_visibility(
        self,
        confidence
    ):
        """
        Check whether the upper body
        is sufficiently visible.
        """

        required = [
            5,  # left shoulder
            6,  # right shoulder
            11, # left hip
            12  # right hip
        ]

        visible = 0

        for index in required:

            if self.is_landmark_valid(
                confidence,
                index
            ):

                visible += 1

        return visible >= 3

    # ======================================================
    # ACTIVITY SMOOTHING
    # ======================================================

    def smooth_activity(
        self,
        activity
    ):
        """
        Prevent activity from changing
        every single frame.
        """

        self.activity_history.append(
            activity
        )

        if len(
            self.activity_history
        ) < 4:

            return activity

        counts = {}

        for item in self.activity_history:

            counts[item] = (
                counts.get(
                    item,
                    0
                ) + 1
            )

        most_common = max(
            counts,
            key=counts.get
        )

        return most_common

    # ======================================================
    # MOVEMENT CALCULATION
    # ======================================================

    def calculate_movement(
        self,
        current_point,
        previous_point,
        body_scale
    ):
        """
        Calculate normalized movement.

        Normalization by body scale prevents
        camera distance from affecting the
        movement threshold too much.
        """

        try:

            if previous_point is None:
                return 0.0

            dx = (
                current_point[0] -
                previous_point[0]
            )

            dy = (
                current_point[1] -
                previous_point[1]
            )

            movement = np.sqrt(
                dx ** 2 +
                dy ** 2
            )

            normalized = (
                movement /
                max(body_scale, 1.0)
            )

            return float(
                normalized
            )

        except Exception:

            return 0.0

    # ======================================================
    # MAIN DETECTION
    # ======================================================

    def detect(
        self,
        keypoints,
        landmark_confidence
    ):
        """
        Detect the current human activity.

        Parameters
        ----------
        keypoints:
            YOLO 17 body keypoints.

        landmark_confidence:
            YOLO keypoint confidence values.

        Returns
        -------
        activity, confidence
        """

        self.frame_count += 1

        try:

            # ------------------------------------------------
            # NO PERSON
            # ------------------------------------------------

            if keypoints is None:

                self.reset()

                return (
                    self.NO_PERSON,
                    0.0
                )

            # ------------------------------------------------
            # NO CONFIDENCE INFORMATION
            # ------------------------------------------------

            if landmark_confidence is None:

                return (
                    self.BODY_NOT_VISIBLE,
                    0.0
                )

            # ------------------------------------------------
            # FULL BODY CHECK
            # ------------------------------------------------

            full_body_visible = (
                self.check_body_visibility(
                    landmark_confidence
                )
            )

            if not full_body_visible:

                self.current_activity = (
                    self.BODY_NOT_VISIBLE
                )

                self.confidence = 0.0

                # Do not keep old activity.
                self.activity_history.clear()

                return (
                    self.current_activity,
                    self.confidence
                )

            # ------------------------------------------------
            # EXTRACT LANDMARKS
            # ------------------------------------------------

            left_shoulder = keypoints[5]

            right_shoulder = keypoints[6]

            left_hip = keypoints[11]

            right_hip = keypoints[12]

            left_knee = keypoints[13]

            right_knee = keypoints[14]

            left_ankle = keypoints[15]

            right_ankle = keypoints[16]

            # ------------------------------------------------
            # MIDPOINTS
            # ------------------------------------------------

            shoulder_center = (
                calculate_midpoint(
                    left_shoulder,
                    right_shoulder
                )
            )

            hip_center = (
                calculate_midpoint(
                    left_hip,
                    right_hip
                )
            )

            knee_center = (
                calculate_midpoint(
                    left_knee,
                    right_knee
                )
            )

            ankle_center = (
                calculate_midpoint(
                    left_ankle,
                    right_ankle
                )
            )

            # ------------------------------------------------
            # BODY SCALE
            # ------------------------------------------------

            torso_length = np.linalg.norm(
                np.array(
                    hip_center
                )
                -
                np.array(
                    shoulder_center
                )
            )

            leg_length = np.linalg.norm(
                np.array(
                    ankle_center
                )
                -
                np.array(
                    hip_center
                )
            )

            body_scale = max(
                torso_length + leg_length,
                1.0
            )

            # ------------------------------------------------
            # TORSO ORIENTATION
            # ------------------------------------------------

            torso_vertical_angle = (
                calculate_vertical_angle(
                    shoulder_center,
                    hip_center
                )
            )

            torso_horizontal_angle = (
                calculate_horizontal_angle(
                    shoulder_center,
                    hip_center
                )
            )

            # ------------------------------------------------
            # KNEE ANGLES
            # ------------------------------------------------

            left_knee_angle = (
                calculate_angle(
                    left_hip,
                    left_knee,
                    left_ankle
                )
            )

            right_knee_angle = (
                calculate_angle(
                    right_hip,
                    right_knee,
                    right_ankle
                )
            )

            average_knee_angle = (
                left_knee_angle +
                right_knee_angle
            ) / 2.0

            # ------------------------------------------------
            # MOVEMENT
            # ------------------------------------------------

            hip_movement = (
                self.calculate_movement(
                    hip_center,
                    self.previous_hip,
                    body_scale
                )
            )

            ankle_movement = 0.0

            if self.previous_ankle is not None:

                ankle_movement = (
                    self.calculate_movement(
                        ankle_center,
                        self.previous_ankle,
                        body_scale
                    )
                )

            self.previous_hip = (
                hip_center
            )

            self.previous_shoulder = (
                shoulder_center
            )

            self.previous_ankle = (
                ankle_center
            )

            movement = (
                hip_movement * 0.6
                +
                ankle_movement * 0.4
            )

            self.movement_history.append(
                movement
            )

            average_movement = (
                np.mean(
                    self.movement_history
                )
                if self.movement_history
                else 0.0
            )

            # =================================================
            # LYING DOWN DETECTION
            # =================================================
            #
            # A standing person's shoulder-hip
            # line is approximately vertical.
            #
            # A lying person's shoulder-hip
            # line is approximately horizontal.
            #
            # We use BOTH orientation and body
            # geometry rather than movement alone.
            # =================================================

            if (
                torso_vertical_angle > 55
                and
                torso_horizontal_angle < 35
            ):

                activity = self.LYING

                confidence = 0.90

                self.current_activity = (
                    self.smooth_activity(
                        activity
                    )
                )

                self.confidence = confidence

                return (
                    self.current_activity,
                    self.confidence
                )

            # =================================================
            # SITTING DETECTION
            # =================================================

            knee_is_bent = (
                average_knee_angle < 130
            )

            hips_close_to_knees = (
                abs(
                    hip_center[1] -
                    knee_center[1]
                )
                <
                body_scale * 0.35
            )

            if (
                knee_is_bent
                and
                hips_close_to_knees
            ):

                activity = self.SITTING

                confidence = 0.85

                self.current_activity = (
                    self.smooth_activity(
                        activity
                    )
                )

                self.confidence = confidence

                return (
                    self.current_activity,
                    self.confidence
                )

            # =================================================
            # WALKING / RUNNING
            # =================================================
            #
            # Movement must be sustained over several
            # frames before changing activity.
            # =================================================

            if len(
                self.movement_history
            ) >= 5:

                if average_movement > 0.14:

                    activity = self.RUNNING

                    confidence = 0.86

                    self.current_activity = (
                        self.smooth_activity(
                            activity
                        )
                    )

                    self.confidence = (
                        confidence
                    )

                    return (
                        self.current_activity,
                        self.confidence
                    )

                if average_movement > 0.045:

                    activity = self.WALKING

                    confidence = 0.84

                    self.current_activity = (
                        self.smooth_activity(
                            activity
                        )
                    )

                    self.confidence = (
                        confidence
                    )

                    return (
                        self.current_activity,
                        self.confidence
                    )

            # =================================================
            # STANDING
            # =================================================

            if (
                torso_vertical_angle < 35
                and
                average_knee_angle > 145
            ):

                activity = self.STANDING

                confidence = 0.90

                self.current_activity = (
                    self.smooth_activity(
                        activity
                    )
                )

                self.confidence = (
                    confidence
                )

                return (
                    self.current_activity,
                    self.confidence
                )

            # =================================================
            # FALLBACK
            # =================================================

            # If the body is visible but the geometry
            # doesn't strongly match an activity, don't
            # randomly call it standing.
            activity = (
                self.current_activity
                if self.current_activity
                not in [
                    self.NO_PERSON,
                    self.BODY_NOT_VISIBLE
                ]
                else self.STANDING
            )

            confidence = 0.60

            self.current_activity = (
                self.smooth_activity(
                    activity
                )
            )

            self.confidence = confidence

            return (
                self.current_activity,
                self.confidence
            )

        except Exception as error:

            print(
                "Activity detection error:",
                error
            )

            return (
                self.BODY_NOT_VISIBLE,
                0.0
            )

    # ======================================================
    # GET CURRENT ACTIVITY
    # ======================================================

    def get_activity(self):

        return self.current_activity

    # ======================================================
    # GET CONFIDENCE
    # ======================================================

    def get_confidence(self):

        return self.confidence