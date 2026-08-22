import os

# ============================================
# CPU OPTIMIZATION
# ============================================

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"

from pathlib import Path
import time

import cv2
import numpy as np

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "yolo11n-pose.pt"


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="AI Human Activity Recognition",
    version="5.0.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(BASE_DIR / "static")
    ),
    name="static"
)


# ============================================================
# HTML TEMPLATES
# ============================================================

templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)


# ============================================================
# EXERCISE NAMES
# ============================================================

EXERCISE_NAMES = {
    "squat": "Squat",
    "bicep_curl": "Bicep Curl",
    "shoulder_press": "Shoulder Press"
}


# ============================================================
# LOAD YOLO MODEL ONCE
# ============================================================

model = None

try:

    if MODEL_PATH.exists():

        print("=" * 60)
        print("LOADING YOLO11 POSE MODEL")
        print("=" * 60)

        model = YOLO(
            str(MODEL_PATH)
        )

        # Force CPU for Render
        try:
            model.to("cpu")
        except Exception:
            pass

        print("YOLO11 POSE MODEL LOADED")
        print("Device: CPU")
        print(f"Model: {MODEL_PATH}")
        print("=" * 60)

    else:

        print("=" * 60)
        print("ERROR: YOLO MODEL NOT FOUND")
        print(f"Expected file: {MODEL_PATH}")
        print("=" * 60)

except Exception as error:

    print("=" * 60)
    print("ERROR LOADING YOLO MODEL")
    print(error)
    print("=" * 60)


# ============================================================
# HOME PAGE
# ============================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
async def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "message": "AI Human Activity Recognition is running"
    }


# ============================================================
# GET KEYPOINT
# ============================================================

def get_point(
    keypoints,
    index
):

    try:

        point = keypoints[index]

        x = float(point[0])
        y = float(point[1])

        confidence = (
            float(point[2])
            if len(point) >= 3
            else 1.0
        )

        return (
            np.array(
                [x, y],
                dtype=np.float32
            ),
            confidence
        )

    except Exception:

        return None, 0.0


# ============================================================
# DISTANCE
# ============================================================

def distance(
    point_a,
    point_b
):

    try:

        return float(
            np.linalg.norm(
                np.array(point_a)
                -
                np.array(point_b)
            )
        )

    except Exception:

        return 0.0


# ============================================================
# ANGLE
# ============================================================

def calculate_angle(
    point_a,
    point_b,
    point_c
):

    try:

        a = np.array(
            point_a,
            dtype=np.float32
        )

        b = np.array(
            point_b,
            dtype=np.float32
        )

        c = np.array(
            point_c,
            dtype=np.float32
        )

        ba = a - b
        bc = c - b

        magnitude_ba = np.linalg.norm(ba)
        magnitude_bc = np.linalg.norm(bc)

        if (
            magnitude_ba < 1e-6
            or
            magnitude_bc < 1e-6
        ):

            return None

        cosine = (
            np.dot(ba, bc)
            /
            (
                magnitude_ba
                *
                magnitude_bc
            )
        )

        cosine = np.clip(
            cosine,
            -1.0,
            1.0
        )

        angle = np.degrees(
            np.arccos(cosine)
        )

        return float(
            round(angle, 2)
        )

    except Exception:

        return None


# ============================================================
# ACTIVITY CLASSIFIER
# ============================================================

def classify_activity(
    keypoints
):

    try:

        if (
            keypoints is None
            or
            len(keypoints) < 17
        ):

            return "UNKNOWN", 0


        # ----------------------------------------------------
        # COCO KEYPOINTS
        # ----------------------------------------------------
        #
        # 5  = Left Shoulder
        # 6  = Right Shoulder
        # 11 = Left Hip
        # 12 = Right Hip
        # 13 = Left Knee
        # 14 = Right Knee
        # 15 = Left Ankle
        # 16 = Right Ankle
        #
        # ----------------------------------------------------

        left_shoulder, ls = get_point(
            keypoints,
            5
        )

        right_shoulder, rs = get_point(
            keypoints,
            6
        )

        left_hip, lh = get_point(
            keypoints,
            11
        )

        right_hip, rh = get_point(
            keypoints,
            12
        )

        left_knee, lk = get_point(
            keypoints,
            13
        )

        right_knee, rk = get_point(
            keypoints,
            14
        )

        left_ankle, la = get_point(
            keypoints,
            15
        )

        right_ankle, ra = get_point(
            keypoints,
            16
        )


        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        confidences = [
            ls,
            rs,
            lh,
            rh,
            lk,
            rk,
            la,
            ra
        ]

        reliable_points = sum(
            confidence >= 0.30
            for confidence in confidences
        )

        if reliable_points < 6:

            return "UNKNOWN", 20


        avg_confidence = (
            sum(confidences)
            /
            len(confidences)
        )


        # ----------------------------------------------------
        # BODY CENTERS
        # ----------------------------------------------------

        shoulder = (
            left_shoulder
            +
            right_shoulder
        ) / 2.0

        hip = (
            left_hip
            +
            right_hip
        ) / 2.0

        knee = (
            left_knee
            +
            right_knee
        ) / 2.0

        ankle = (
            left_ankle
            +
            right_ankle
        ) / 2.0


        # ----------------------------------------------------
        # BODY DIMENSIONS
        # ----------------------------------------------------

        torso_length = distance(
            shoulder,
            hip
        )

        leg_length = distance(
            hip,
            ankle
        )

        shoulder_width = distance(
            left_shoulder,
            right_shoulder
        )

        if torso_length < 5:

            return "UNKNOWN", 20


        scale = max(
            torso_length,
            leg_length,
            1
        )


        # ----------------------------------------------------
        # TORSO ORIENTATION
        # ----------------------------------------------------

        dx = (
            hip[0]
            -
            shoulder[0]
        )

        dy = (
            hip[1]
            -
            shoulder[1]
        )

        torso_angle = abs(
            np.degrees(
                np.arctan2(
                    dy,
                    dx
                )
            )
        )

        torso_from_horizontal = min(
            torso_angle,
            180 - torso_angle
        )


        # ----------------------------------------------------
        # KNEE ANGLES
        # ----------------------------------------------------

        left_knee_angle = calculate_angle(
            left_hip,
            left_knee,
            left_ankle
        )

        right_knee_angle = calculate_angle(
            right_hip,
            right_knee,
            right_ankle
        )

        knee_angles = [
            angle
            for angle in [
                left_knee_angle,
                right_knee_angle
            ]
            if angle is not None
        ]

        if not knee_angles:

            return "UNKNOWN", 20

        average_knee_angle = (
            sum(knee_angles)
            /
            len(knee_angles)
        )


        # ----------------------------------------------------
        # HIP ANGLES
        # ----------------------------------------------------

        left_hip_angle = calculate_angle(
            left_shoulder,
            left_hip,
            left_knee
        )

        right_hip_angle = calculate_angle(
            right_shoulder,
            right_hip,
            right_knee
        )

        hip_angles = [
            angle
            for angle in [
                left_hip_angle,
                right_hip_angle
            ]
            if angle is not None
        ]

        average_hip_angle = (
            sum(hip_angles)
            /
            len(hip_angles)
            if hip_angles
            else None
        )


        # ----------------------------------------------------
        # NORMALIZED FEATURES
        # ----------------------------------------------------

        hip_knee_vertical = (
            abs(
                hip[1]
                -
                knee[1]
            )
            /
            scale
        )

        knee_ankle_vertical = (
            abs(
                knee[1]
                -
                ankle[1]
            )
            /
            scale
        )

        hip_ankle_vertical = (
            abs(
                hip[1]
                -
                ankle[1]
            )
            /
            scale
        )


        # ====================================================
        # LYING DOWN
        # ====================================================

        horizontal_span = (
            max(
                left_shoulder[0],
                right_shoulder[0],
                left_hip[0],
                right_hip[0]
            )
            -
            min(
                left_shoulder[0],
                right_shoulder[0],
                left_hip[0],
                right_hip[0]
            )
        )

        vertical_span = (
            max(
                left_shoulder[1],
                right_shoulder[1],
                left_hip[1],
                right_hip[1]
            )
            -
            min(
                left_shoulder[1],
                right_shoulder[1],
                left_hip[1],
                right_hip[1]
            )
        )

        horizontal_ratio = (
            horizontal_span
            /
            max(
                vertical_span,
                1
            )
        )

        lying_score = 0

        if torso_from_horizontal < 35:
            lying_score += 2

        if horizontal_ratio > 1.7:
            lying_score += 2

        if shoulder_width > torso_length * 0.8:
            lying_score += 1

        if lying_score >= 3:

            confidence = min(
                96,
                int(
                    60
                    +
                    avg_confidence * 35
                )
            )

            return (
                "LYING DOWN",
                confidence
            )


        # ====================================================
        # SITTING
        # ====================================================

        sitting_score = 0

        # Torso is reasonably upright
        if torso_from_horizontal > 55:
            sitting_score += 2

        # Knees bent
        if (
            65
            <=
            average_knee_angle
            <=
            140
        ):

            sitting_score += 2

        # Hips bent
        if (
            average_hip_angle is not None
            and
            average_hip_angle < 145
        ):

            sitting_score += 1

        # Hip and knee are vertically close
        if hip_knee_vertical < 0.38:

            sitting_score += 3

        elif hip_knee_vertical < 0.50:

            sitting_score += 1

        # Lower leg visible
        if knee_ankle_vertical > 0.15:

            sitting_score += 1

        # Not fully extended
        if hip_ankle_vertical < 1.15:

            sitting_score += 1


        if (
            sitting_score >= 6
            and
            torso_from_horizontal > 55
            and
            hip_knee_vertical < 0.55
        ):

            confidence = min(
                96,
                int(
                    58
                    +
                    avg_confidence * 38
                )
            )

            return (
                "SITTING",
                confidence
            )


        # ====================================================
        # SQUATTING
        # ====================================================

        squat_score = 0

        # Strongly bent knees
        if average_knee_angle < 105:

            squat_score += 3

        elif average_knee_angle < 120:

            squat_score += 2

        elif average_knee_angle < 135:

            squat_score += 1


        # Torso is not horizontal
        if torso_from_horizontal > 50:

            squat_score += 1


        # Hip above knee
        hip_above_knee = (
            hip[1]
            <
            knee[1]
        )


        if (
            hip_above_knee
            and
            hip_knee_vertical > 0.30
        ):

            squat_score += 3

        elif hip_knee_vertical > 0.45:

            squat_score += 2


        # Bent hips
        if (
            average_hip_angle is not None
            and
            average_hip_angle < 135
        ):

            squat_score += 1


        # Lower legs visible
        if knee_ankle_vertical > 0.15:

            squat_score += 1


        if (
            squat_score >= 6
            and
            torso_from_horizontal > 50
            and
            hip_knee_vertical > 0.28
        ):

            confidence = min(
                95,
                int(
                    55
                    +
                    avg_confidence * 40
                )
            )

            return (
                "SQUATTING",
                confidence
            )


        # ====================================================
        # STANDING
        # ====================================================

        standing_score = 0

        # Upright torso
        if torso_from_horizontal > 60:

            standing_score += 2


        # Straight knees
        if average_knee_angle >= 150:

            standing_score += 3

        elif average_knee_angle >= 140:

            standing_score += 2


        # Straight hips
        if (
            average_hip_angle is not None
            and
            average_hip_angle >= 145
        ):

            standing_score += 1


        # Vertical extension
        if hip_ankle_vertical > 1.0:

            standing_score += 1


        # Lower legs visible
        if knee_ankle_vertical > 0.20:

            standing_score += 1


        if standing_score >= 5:

            confidence = min(
                96,
                int(
                    60
                    +
                    avg_confidence * 35
                )
            )

            return (
                "STANDING",
                confidence
            )


        # ====================================================
        # SECONDARY STANDING
        # ====================================================

        if (
            torso_from_horizontal > 60
            and
            average_knee_angle >= 135
            and
            hip_knee_vertical > 0.35
        ):

            confidence = min(
                88,
                int(
                    52
                    +
                    avg_confidence * 35
                )
            )

            return (
                "STANDING",
                confidence
            )


        return "UNKNOWN", 30


    except Exception as error:

        print(
            "ACTIVITY CLASSIFIER ERROR:",
            error
        )

        return "UNKNOWN", 0


# ============================================================
# EXERCISE ANGLE
# ============================================================

def calculate_exercise_angle(
    keypoints,
    exercise
):

    try:

        # Squat:
        # Left Hip -> Left Knee -> Left Ankle

        if exercise == "squat":

            return calculate_angle(
                keypoints[11],
                keypoints[13],
                keypoints[15]
            )


        # Bicep curl:
        # Left Shoulder -> Left Elbow -> Left Wrist

        if exercise == "bicep_curl":

            return calculate_angle(
                keypoints[5],
                keypoints[7],
                keypoints[9]
            )


        # Shoulder press:
        # Left Hip -> Left Shoulder -> Left Elbow

        if exercise == "shoulder_press":

            return calculate_angle(
                keypoints[11],
                keypoints[5],
                keypoints[7]
            )


        return None


    except Exception:

        return None


# ============================================================
# FEEDBACK
# ============================================================

def get_feedback(
    exercise,
    angle,
    activity
):

    # --------------------------------------------------------
    # UNKNOWN
    # --------------------------------------------------------

    if activity == "UNKNOWN":

        return (
            "Make sure your full body is clearly visible "
            "and stand in front of the camera."
        )


    # --------------------------------------------------------
    # LYING
    # --------------------------------------------------------

    if activity == "LYING DOWN":

        return (
            "You are lying down. "
            "Please stand or sit before exercising."
        )


    # --------------------------------------------------------
    # SITTING
    # --------------------------------------------------------

    if activity == "SITTING":

        if exercise == "squat":

            return (
                "You are sitting. "
                "Stand up before starting squats."
            )

        return (
            "You are sitting. "
            "Maintain a stable posture."
        )


    # --------------------------------------------------------
    # SQUAT
    # --------------------------------------------------------

    if exercise == "squat":

        if activity == "SQUATTING":

            if angle is not None:

                if angle < 90:

                    return (
                        "Excellent squat depth!"
                    )

                if angle < 120:

                    return (
                        "Good squat. "
                        "Try going slightly lower."
                    )

            return (
                "Good squat movement."
            )


        if activity == "STANDING":

            return (
                "Ready. "
                "Bend your knees and lower your hips."
            )


    # --------------------------------------------------------
    # BICEP CURL
    # --------------------------------------------------------

    if exercise == "bicep_curl":

        if angle is None:

            return (
                "Keep your arm clearly visible."
            )

        if angle < 70:

            return (
                "Great curl! "
                "Your arm is well flexed."
            )

        if angle < 110:

            return (
                "Keep curling your arm."
            )

        return (
            "Start with your arm extended."
        )


    # --------------------------------------------------------
    # SHOULDER PRESS
    # --------------------------------------------------------

    if exercise == "shoulder_press":

        if angle is None:

            return (
                "Keep your arm clearly visible."
            )

        if angle > 150:

            return (
                "Excellent shoulder press!"
            )

        if angle > 120:

            return (
                "Keep raising your arm."
            )

        return (
            "Raise your arm higher."
        )


    return (
        "Keep your body visible and continue."
    )


# ============================================================
# RESIZE IMAGE FOR FAST CPU INFERENCE
# ============================================================

def resize_for_inference(
    image,
    max_size=640
):

    try:

        height, width = image.shape[:2]

        largest_dimension = max(
            height,
            width
        )

        if largest_dimension <= max_size:

            return image


        scale = (
            max_size
            /
            largest_dimension
        )

        new_width = int(
            width * scale
        )

        new_height = int(
            height * scale
        )

        resized = cv2.resize(
            image,
            (
                new_width,
                new_height
            ),
            interpolation=cv2.INTER_AREA
        )

        return resized


    except Exception:

        return image


# ============================================================
# PROCESS IMAGE
# ============================================================

def process_image(
    image,
    exercise
):

    start_time = time.perf_counter()


    result_data = {

        "activity": "UNKNOWN",

        "activity_confidence": 0,

        "exercise":
            EXERCISE_NAMES.get(
                exercise,
                exercise
            ),

        "confidence": 0,

        "repetitions": 0,

        "angle": None,

        "person_detected": False,

        "feedback":
            "No person detected.",

        "keypoints": [],

        "processing_time": 0
    }


    # ========================================================
    # CHECK MODEL
    # ========================================================

    if model is None:

        result_data["feedback"] = (
            "YOLO model is not loaded."
        )

        return result_data


    try:

        # ====================================================
        # RESIZE
        # ====================================================

        image = resize_for_inference(
            image,
            max_size=640
        )


        # ====================================================
        # YOLO POSE INFERENCE
        # ====================================================

        results = model.predict(

            source=image,

            imgsz=320,

            conf=0.35,

            iou=0.45,

            device="cpu",

            verbose=False,

            max_det=1
        )


        # ====================================================
        # CHECK RESULT
        # ====================================================

        if not results:

            result_data["feedback"] = (
                "No person detected."
            )

            return result_data


        result = results[0]


        # ====================================================
        # CHECK KEYPOINTS
        # ====================================================

        if (
            result.keypoints is None
            or
            result.keypoints.data is None
            or
            len(result.keypoints.data) == 0
        ):

            result_data["feedback"] = (
                "No human pose detected. "
                "Make sure your full body is visible."
            )

            return result_data


        # ====================================================
        # PERSON SELECTION
        # ====================================================

        person_index = 0


        # ====================================================
        # GET KEYPOINTS
        # ====================================================

        person_keypoints = (
            result
            .keypoints
            .data[
                person_index
            ]
            .cpu()
            .numpy()
        )


        points = []


        for point in person_keypoints:

            points.append(
                [
                    float(point[0]),
                    float(point[1]),
                    float(point[2])
                ]
            )


        result_data["keypoints"] = points

        result_data["person_detected"] = True


        # ====================================================
        # PERSON CONFIDENCE
        # ====================================================

        if result.boxes is not None:

            if len(result.boxes.conf) > person_index:

                confidence = float(
                    result
                    .boxes
                    .conf[
                        person_index
                    ]
                    .cpu()
                    .item()
                )

                result_data["confidence"] = round(
                    confidence * 100,
                    2
                )


        # ====================================================
        # ACTIVITY
        # ====================================================

        activity, activity_confidence = (
            classify_activity(
                points
            )
        )


        result_data["activity"] = activity

        result_data[
            "activity_confidence"
        ] = activity_confidence


        # ====================================================
        # XY POINTS
        # ====================================================

        xy_points = [

            [
                point[0],
                point[1]
            ]

            for point in points
        ]


        # ====================================================
        # EXERCISE ANGLE
        # ====================================================

        angle = calculate_exercise_angle(
            xy_points,
            exercise
        )


        result_data["angle"] = angle


        # ====================================================
        # FEEDBACK
        # ====================================================

        result_data["feedback"] = get_feedback(

            exercise,

            angle,

            activity
        )


        # ====================================================
        # PROCESSING TIME
        # ====================================================

        result_data["processing_time"] = round(
            time.perf_counter()
            -
            start_time,
            3
        )


        print(
            f"Analysis completed in "
            f"{result_data['processing_time']} seconds | "
            f"Activity: {activity}"
        )


        return result_data


    except Exception as error:

        print("=" * 60)
        print("POSE PROCESSING ERROR")
        print(error)
        print("=" * 60)


        result_data["feedback"] = (
            "Unable to process this image."
        )

        result_data["processing_time"] = round(
            time.perf_counter()
            -
            start_time,
            3
        )

        return result_data


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/analyze-image")
async def analyze_image(

    file: UploadFile = File(...),

    exercise: str = Form("squat")

):

    request_start = time.perf_counter()

    try:

        # ====================================================
        # READ IMAGE
        # ====================================================

        image_bytes = await file.read()


        if not image_bytes:

            return {
                "activity": "UNKNOWN",
                "activity_confidence": 0,
                "exercise":
                    EXERCISE_NAMES.get(
                        exercise,
                        exercise
                    ),
                "confidence": 0,
                "repetitions": 0,
                "angle": None,
                "person_detected": False,
                "feedback": "Empty image.",
                "processing_time": 0
            }


        # ====================================================
        # DECODE IMAGE
        # ====================================================

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )


        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )


        if image is None:

            return {
                "activity": "UNKNOWN",
                "activity_confidence": 0,
                "exercise":
                    EXERCISE_NAMES.get(
                        exercise,
                        exercise
                    ),
                "confidence": 0,
                "repetitions": 0,
                "angle": None,
                "person_detected": False,
                "feedback": "Invalid image.",
                "processing_time": 0
            }


        # ====================================================
        # PROCESS
        # ====================================================

        result = process_image(
            image,
            exercise
        )


        result[
            "total_request_time"
        ] = round(
            time.perf_counter()
            -
            request_start,
            3
        )


        return result


    except Exception as error:

        print("=" * 60)
        print("IMAGE ANALYSIS ERROR")
        print(error)
        print("=" * 60)


        return {

            "activity": "ERROR",

            "activity_confidence": 0,

            "exercise":
                EXERCISE_NAMES.get(
                    exercise,
                    exercise
                ),

            "confidence": 0,

            "repetitions": 0,

            "angle": None,

            "person_detected": False,

            "feedback":
                "Unable to analyze image.",

            "processing_time":
                round(
                    time.perf_counter()
                    -
                    request_start,
                    3
                )
        }


# ============================================================
# LIVE FRAME ANALYSIS
# ============================================================

@app.post("/analyze-frame")
async def analyze_frame(

    file: UploadFile = File(...),

    exercise: str = Form("squat")

):

    try:

        frame_bytes = await file.read()


        frame_array = np.frombuffer(
            frame_bytes,
            dtype=np.uint8
        )


        frame = cv2.imdecode(
            frame_array,
            cv2.IMREAD_COLOR
        )


        if frame is None:

            return {

                "activity": "UNKNOWN",

                "activity_confidence": 0,

                "exercise":
                    EXERCISE_NAMES.get(
                        exercise,
                        exercise
                    ),

                "confidence": 0,

                "repetitions": 0,

                "angle": None,

                "person_detected": False,

                "feedback":
                    "Invalid camera frame."
            }


        return process_image(
            frame,
            exercise
        )


    except Exception as error:

        print("=" * 60)
        print("FRAME ANALYSIS ERROR")
        print(error)
        print("=" * 60)


        return {

            "activity": "ERROR",

            "activity_confidence": 0,

            "exercise":
                EXERCISE_NAMES.get(
                    exercise,
                    exercise
                ),

            "confidence": 0,

            "repetitions": 0,

            "angle": None,

            "person_detected": False,

            "feedback":
                "Unable to analyze camera frame."
        }