from ultralytics import YOLO
import numpy as np


class PoseDetector:

    def __init__(
        self,
        model_path="yolo11n-pose.pt",
        confidence=0.35
    ):
        """
        Initialize YOLO Pose detector.
        """

        self.model = YOLO(model_path)

        self.confidence = confidence

        self.landmark_names = {
            0: "Nose",
            1: "Left Eye",
            2: "Right Eye",
            3: "Left Ear",
            4: "Right Ear",
            5: "Left Shoulder",
            6: "Right Shoulder",
            7: "Left Elbow",
            8: "Right Elbow",
            9: "Left Wrist",
            10: "Right Wrist",
            11: "Left Hip",
            12: "Right Hip",
            13: "Left Knee",
            14: "Right Knee",
            15: "Left Ankle",
            16: "Right Ankle"
        }

    def process_frame(self, frame):
        """
        Run YOLO Pose on a frame.

        Returns
        -------
        annotated_frame
        keypoints
        landmark_confidence
        person_confidence
        """

        keypoints = None
        landmark_confidence = None
        person_confidence = 0.0

        try:

            results = self.model(
                frame,
                verbose=False,
                conf=self.confidence
            )

            if len(results) == 0:
                return frame, None, None, 0.0

            result = results[0]

            annotated_frame = result.plot()

            if result.boxes is None or len(result.boxes) == 0:
                return annotated_frame, None, None, 0.0

            # Select the person with highest detection confidence.
            box_conf = result.boxes.conf.cpu().numpy()

            best_index = int(np.argmax(box_conf))

            person_confidence = float(box_conf[best_index])

            if result.keypoints is None:
                return annotated_frame, None, None, person_confidence

            xy = result.keypoints.xy.cpu().numpy()

            if len(xy) == 0:
                return annotated_frame, None, None, person_confidence

            # Select same person as the best bounding box.
            keypoints = xy[best_index]

            # YOLO keypoint confidence.
            if result.keypoints.conf is not None:

                conf_values = (
                    result.keypoints.conf.cpu().numpy()
                )

                if len(conf_values) > best_index:
                    landmark_confidence = conf_values[
                        best_index
                    ]

            return (
                annotated_frame,
                keypoints,
                landmark_confidence,
                person_confidence
            )

        except Exception as error:

            print(
                f"Pose detection error: {error}"
            )

            return (
                frame,
                None,
                None,
                0.0
            )

    def is_landmark_visible(
        self,
        landmark_confidence,
        index,
        threshold=0.45
    ):
        """
        Check whether a landmark is reliable.
        """

        try:

            if landmark_confidence is None:
                return False

            if index >= len(landmark_confidence):
                return False

            return (
                float(landmark_confidence[index])
                >= threshold
            )

        except Exception:
            return False

    def visible_landmark_count(
        self,
        landmark_confidence,
        threshold=0.45
    ):
        """
        Count reliable body landmarks.
        """

        if landmark_confidence is None:
            return 0

        try:

            return int(
                np.sum(
                    landmark_confidence >= threshold
                )
            )

        except Exception:
            return 0

    def get_landmark_name(self, index):
        return self.landmark_names.get(
            index,
            "Unknown"
        )