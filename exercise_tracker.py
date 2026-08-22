from angle_utils import calculate_angle


class ExerciseTracker:

    SQUAT = "Squat"
    CURL = "Bicep Curl"
    PRESS = "Shoulder Press"

    def __init__(self):

        self.exercise = "AUTO"

        self.counter = 0

        self.stage = "READY"

        self.feedback = (
            "Ready - select an exercise"
        )

        self.last_angle = 0.0

        self.detected_exercise = "NONE"

    def reset(self):

        self.counter = 0

        self.stage = "READY"

        self.feedback = "Ready"

        self.last_angle = 0.0

        self.detected_exercise = "NONE"

    def set_exercise(self, exercise):

        self.exercise = exercise

        self.reset()

        if exercise == "AUTO":
            self.feedback = (
                "Automatic exercise detection enabled"
            )
        else:
            self.feedback = (
                f"{exercise} selected"
            )

    def _valid(
        self,
        confidence,
        index,
        threshold=0.45
    ):

        try:
            return (
                confidence is not None
                and float(confidence[index])
                >= threshold
            )

        except Exception:
            return False

    def process(
        self,
        keypoints,
        confidence
    ):

        try:

            if keypoints is None:
                return (
                    self.counter,
                    self.detected_exercise,
                    "Body not detected"
                )

            # Automatic mode
            if self.exercise == "AUTO":

                detected = (
                    self._auto_detect(
                        keypoints,
                        confidence
                    )
                )

                if detected == self.SQUAT:

                    return self._track_squat(
                        keypoints,
                        confidence
                    )

                if detected == self.CURL:

                    return self._track_curl(
                        keypoints,
                        confidence
                    )

                if detected == self.PRESS:

                    return self._track_press(
                        keypoints,
                        confidence
                    )

                return (
                    self.counter,
                    self.detected_exercise,
                    self.feedback
                )

            if self.exercise == self.SQUAT:

                return self._track_squat(
                    keypoints,
                    confidence
                )

            if self.exercise == self.CURL:

                return self._track_curl(
                    keypoints,
                    confidence
                )

            if self.exercise == self.PRESS:

                return self._track_press(
                    keypoints,
                    confidence
                )

        except Exception as error:

            print(
                f"Exercise tracking error: {error}"
            )

        return (
            self.counter,
            self.detected_exercise,
            self.feedback
        )

    def _auto_detect(
        self,
        kp,
        confidence
    ):

        # Check squat landmarks.
        squat_ready = all(
            self._valid(confidence, index)
            for index in [
                11, 13, 15
            ]
        )

        # Check arm landmarks.
        curl_ready = all(
            self._valid(confidence, index)
            for index in [
                5, 7, 9
            ]
        )

        press_ready = all(
            self._valid(confidence, index)
            for index in [
                11, 5, 7
            ]
        )

        if squat_ready:

            squat_angle = calculate_angle(
                kp[11],
                kp[13],
                kp[15]
            )

            if squat_angle < 125:

                return self.SQUAT

        if curl_ready:

            curl_angle = calculate_angle(
                kp[5],
                kp[7],
                kp[9]
            )

            if curl_angle < 100:

                return self.CURL

        if press_ready:

            press_angle = calculate_angle(
                kp[11],
                kp[5],
                kp[7]
            )

            if press_angle > 145:

                return self.PRESS

        return "NONE"

    # ======================================================
    # SQUAT
    # ======================================================

    def _track_squat(
        self,
        kp,
        confidence
    ):

        self.detected_exercise = self.SQUAT

        if not all(
            self._valid(confidence, index)
            for index in [
                11, 13, 15
            ]
        ):

            self.feedback = (
                "Show your full lower body"
            )

            return (
                self.counter,
                self.detected_exercise,
                self.feedback
            )

        angle = calculate_angle(
            kp[11],
            kp[13],
            kp[15]
        )

        self.last_angle = angle

        if angle > 160:

            if self.stage == "DOWN":

                self.counter += 1

            self.stage = "UP"

            self.feedback = (
                "Stand upright"
            )

        elif angle < 95:

            if self.stage == "UP":

                self.stage = "DOWN"

            self.feedback = (
                "Good depth!"
            )

        elif angle < 120:

            self.feedback = (
                "Go a little lower"
            )

        else:

            self.feedback = (
                "Continue the movement"
            )

        return (
            self.counter,
            self.detected_exercise,
            self.feedback
        )

    # ======================================================
    # BICEP CURL
    # ======================================================

    def _track_curl(
        self,
        kp,
        confidence
    ):

        self.detected_exercise = self.CURL

        if not all(
            self._valid(confidence, index)
            for index in [
                5, 7, 9
            ]
        ):

            self.feedback = (
                "Show your shoulder, elbow and wrist"
            )

            return (
                self.counter,
                self.detected_exercise,
                self.feedback
            )

        angle = calculate_angle(
            kp[5],
            kp[7],
            kp[9]
        )

        self.last_angle = angle

        if angle > 155:

            if self.stage == "UP":

                self.counter += 1

            self.stage = "DOWN"

            self.feedback = (
                "Arm extended - curl up"
            )

        elif angle < 55:

            self.stage = "UP"

            self.feedback = (
                "Great curl!"
            )

        elif angle < 85:

            self.feedback = (
                "Almost there"
            )

        else:

            self.feedback = (
                "Curl higher"
            )

        return (
            self.counter,
            self.detected_exercise,
            self.feedback
        )

    # ======================================================
    # SHOULDER PRESS
    # ======================================================

    def _track_press(
        self,
        kp,
        confidence
    ):

        self.detected_exercise = self.PRESS

        if not all(
            self._valid(confidence, index)
            for index in [
                11, 5, 7
            ]
        ):

            self.feedback = (
                "Show your hip, shoulder and elbow"
            )

            return (
                self.counter,
                self.detected_exercise,
                self.feedback
            )

        angle = calculate_angle(
            kp[11],
            kp[5],
            kp[7]
        )

        self.last_angle = angle

        if angle < 80:

            if self.stage == "UP":

                self.counter += 1

            self.stage = "DOWN"

            self.feedback = (
                "Press upward"
            )

        elif angle > 150:

            self.stage = "UP"

            self.feedback = (
                "Good press!"
            )

        elif angle > 120:

            self.feedback = (
                "Extend your arm"
            )

        else:

            self.feedback = (
                "Keep pressing"
            )

        return (
            self.counter,
            self.detected_exercise,
            self.feedback
        )