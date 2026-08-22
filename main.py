import customtkinter as ctk
import cv2
import time

from PIL import Image

from tkinter import filedialog, messagebox

from pose_detector import PoseDetector
from activity_detector import ActivityDetector
from exercise_tracker import ExerciseTracker


# ==========================================================
# APPLICATION SETTINGS
# ==========================================================

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class HARApplication:

    def __init__(self, root):

        self.root = root

        self.root.title(
            "AI Human Activity & Fitness Recognition"
        )

        self.root.geometry(
            "1500x900"
        )

        self.root.minsize(
            1100,
            700
        )

        # --------------------------------------------------
        # AI COMPONENTS
        # --------------------------------------------------

        self.pose_detector = PoseDetector()

        self.activity_detector = (
            ActivityDetector()
        )

        self.exercise_tracker = (
            ExerciseTracker()
        )

        # --------------------------------------------------
        # CAMERA
        # --------------------------------------------------

        self.camera = None

        self.camera_running = False

        # --------------------------------------------------
        # SESSION
        # --------------------------------------------------

        self.session_start = None

        self.total_frames = 0

        self.last_fps_time = time.time()

        self.fps = 0.0

        # --------------------------------------------------
        # UI
        # --------------------------------------------------

        self.create_interface()

        self.root.protocol(
            "WM_DELETE_WINDOW",
            self.close_application
        )

    # ======================================================
    # CREATE USER INTERFACE
    # ======================================================

    def create_interface(self):

        self.root.grid_columnconfigure(
            0,
            weight=0
        )

        self.root.grid_columnconfigure(
            1,
            weight=1
        )

        self.root.grid_rowconfigure(
            0,
            weight=1
        )

        # ==================================================
        # SIDEBAR
        # ==================================================

        sidebar = ctk.CTkFrame(
            self.root,
            corner_radius=0
        )

        sidebar.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        sidebar.grid_rowconfigure(
            11,
            weight=1
        )

        title = ctk.CTkLabel(
            sidebar,
            text="AI FITNESS\nTRACKER",
            font=(
                "Segoe UI",
                26,
                "bold"
            )
        )

        title.grid(
            row=0,
            column=0,
            padx=25,
            pady=(30, 10)
        )

        subtitle = ctk.CTkLabel(
            sidebar,
            text="Human Activity Recognition",
            font=(
                "Segoe UI",
                13
            )
        )

        subtitle.grid(
            row=1,
            column=0,
            padx=20,
            pady=(0, 25)
        )

        # --------------------------------------------------
        # EXERCISE SELECTOR
        # --------------------------------------------------

        exercise_title = ctk.CTkLabel(
            sidebar,
            text="EXERCISE MODE",
            font=(
                "Segoe UI",
                12,
                "bold"
            )
        )

        exercise_title.grid(
            row=2,
            column=0,
            padx=25,
            pady=(5, 5),
            sticky="w"
        )

        self.exercise_menu = ctk.CTkOptionMenu(
            sidebar,
            values=[
                "AUTO",
                "Squat",
                "Bicep Curl",
                "Shoulder Press"
            ],
            command=self.change_exercise,
            width=240
        )

        self.exercise_menu.set(
            "AUTO"
        )

        self.exercise_menu.grid(
            row=3,
            column=0,
            padx=25,
            pady=(0, 15)
        )

        # --------------------------------------------------
        # ACTIVITY
        # --------------------------------------------------

        activity_title = ctk.CTkLabel(
            sidebar,
            text="CURRENT ACTIVITY",
            font=(
                "Segoe UI",
                11,
                "bold"
            )
        )

        activity_title.grid(
            row=4,
            column=0,
            pady=(5, 2)
        )

        self.activity_label = ctk.CTkLabel(
            sidebar,
            text="WAITING",
            font=(
                "Segoe UI",
                23,
                "bold"
            )
        )

        self.activity_label.grid(
            row=5,
            column=0,
            pady=(0, 10)
        )

        # --------------------------------------------------
        # CONFIDENCE
        # --------------------------------------------------

        self.confidence_label = ctk.CTkLabel(
            sidebar,
            text="Confidence: --",
            font=(
                "Segoe UI",
                13
            )
        )

        self.confidence_label.grid(
            row=6,
            column=0,
            pady=5
        )

        # --------------------------------------------------
        # EXERCISE
        # --------------------------------------------------

        self.exercise_label = ctk.CTkLabel(
            sidebar,
            text="Exercise: NONE",
            font=(
                "Segoe UI",
                17,
                "bold"
            )
        )

        self.exercise_label.grid(
            row=7,
            column=0,
            pady=8
        )

        # --------------------------------------------------
        # REP COUNT
        # --------------------------------------------------

        self.rep_label = ctk.CTkLabel(
            sidebar,
            text="0",
            font=(
                "Segoe UI",
                48,
                "bold"
            )
        )

        self.rep_label.grid(
            row=8,
            column=0,
            pady=0
        )

        self.rep_title = ctk.CTkLabel(
            sidebar,
            text="REPETITIONS",
            font=(
                "Segoe UI",
                11,
                "bold"
            )
        )

        self.rep_title.grid(
            row=9,
            column=0,
            pady=(0, 10)
        )

        # --------------------------------------------------
        # FEEDBACK
        # --------------------------------------------------

        self.feedback_label = ctk.CTkLabel(
            sidebar,
            text="Ready to start",
            wraplength=240,
            font=(
                "Segoe UI",
                14
            )
        )

        self.feedback_label.grid(
            row=10,
            column=0,
            padx=25,
            pady=15,
            sticky="n"
        )

        # ==================================================
        # BUTTONS
        # ==================================================

        self.start_button = ctk.CTkButton(
            sidebar,
            text="START CAMERA",
            command=self.start_camera,
            height=40
        )

        self.start_button.grid(
            row=12,
            column=0,
            padx=25,
            pady=5
        )

        self.stop_button = ctk.CTkButton(
            sidebar,
            text="STOP CAMERA",
            command=self.stop_camera,
            height=40
        )

        self.stop_button.grid(
            row=13,
            column=0,
            padx=25,
            pady=5
        )

        self.upload_button = ctk.CTkButton(
            sidebar,
            text="UPLOAD IMAGE",
            command=self.upload_image,
            height=40
        )

        self.upload_button.grid(
            row=14,
            column=0,
            padx=25,
            pady=(5, 25)
        )

        # ==================================================
        # MAIN CONTENT
        # ==================================================

        content = ctk.CTkFrame(
            self.root,
            corner_radius=0
        )

        content.grid(
            row=0,
            column=1,
            sticky="nsew",
            padx=10,
            pady=10
        )

        content.grid_rowconfigure(
            1,
            weight=1
        )

        content.grid_columnconfigure(
            0,
            weight=1
        )

        # Header

        header = ctk.CTkLabel(
            content,
            text="LIVE ANALYSIS",
            font=(
                "Segoe UI",
                22,
                "bold"
            )
        )

        header.grid(
            row=0,
            column=0,
            padx=20,
            pady=(15, 10),
            sticky="w"
        )

        # --------------------------------------------------
        # VIDEO AREA
        # --------------------------------------------------

        self.video_frame = ctk.CTkFrame(
            content,
            corner_radius=12
        )

        self.video_frame.grid(
            row=1,
            column=0,
            sticky="nsew",
            padx=20,
            pady=10
        )

        self.video_frame.grid_rowconfigure(
            0,
            weight=1
        )

        self.video_frame.grid_columnconfigure(
            0,
            weight=1
        )

        self.video_label = ctk.CTkLabel(
            self.video_frame,
            text="CAMERA OFF",
            font=(
                "Segoe UI",
                25,
                "bold"
            )
        )

        self.video_label.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        # --------------------------------------------------
        # STATUS BAR
        # --------------------------------------------------

        status = ctk.CTkFrame(
            content
        )

        status.grid(
            row=2,
            column=0,
            sticky="ew",
            padx=20,
            pady=(10, 20)
        )

        self.status_label = ctk.CTkLabel(
            status,
            text="System ready",
            font=(
                "Segoe UI",
                13
            )
        )

        self.status_label.pack(
            side="left",
            padx=15,
            pady=10
        )

        self.fps_label = ctk.CTkLabel(
            status,
            text="FPS: --",
            font=(
                "Segoe UI",
                13
            )
        )

        self.fps_label.pack(
            side="right",
            padx=15
        )

    # ======================================================
    # EXERCISE CHANGE
    # ======================================================

    def change_exercise(self, value):

        self.exercise_tracker.set_exercise(
            value
        )

        self.rep_label.configure(
            text="0"
        )

        self.exercise_label.configure(
            text="Exercise: NONE"
        )

        self.feedback_label.configure(
            text=self.exercise_tracker.feedback
        )

    # ======================================================
    # START CAMERA
    # ======================================================

    def start_camera(self):

        if self.camera_running:
            return

        self.camera = cv2.VideoCapture(
            0,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():

            self.camera.release()

            self.camera = cv2.VideoCapture(
                0
            )

        if not self.camera.isOpened():

            messagebox.showerror(
                "Camera Error",
                "Unable to open the webcam."
            )

            self.status_label.configure(
                text="Camera unavailable"
            )

            return

        self.camera.set(
            cv2.CAP_PROP_FRAME_WIDTH,
            1280
        )

        self.camera.set(
            cv2.CAP_PROP_FRAME_HEIGHT,
            720
        )

        self.camera_running = True

        self.session_start = time.time()

        self.activity_detector.reset()

        self.exercise_tracker.reset()

        self.status_label.configure(
            text="Camera running"
        )

        self.video_label.configure(
            text=""
        )

        self.update_frame()

    # ======================================================
    # STOP CAMERA
    # ======================================================

    def stop_camera(self):

        self.camera_running = False

        if self.camera is not None:

            self.camera.release()

            self.camera = None

        self.video_label.configure(
            image=None,
            text="CAMERA OFF"
        )

        self.status_label.configure(
            text="Camera stopped"
        )

    # ======================================================
    # UPLOAD IMAGE
    # ======================================================

    def upload_image(self):

        # Stop webcam if it is running.
        if self.camera_running:

            self.stop_camera()

        file_path = filedialog.askopenfilename(
            title="Select Human Activity Image",
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp *.webp"
                ),
                (
                    "JPEG Files",
                    "*.jpg *.jpeg"
                ),
                (
                    "PNG Files",
                    "*.png"
                ),
                (
                    "All Files",
                    "*.*"
                )
            ]
        )

        if not file_path:
            return

        try:

            frame = cv2.imread(
                file_path
            )

            if frame is None:

                messagebox.showerror(
                    "Image Error",
                    "Unable to read the selected image."
                )

                return

            # --------------------------------------------------
            # YOLO POSE
            # --------------------------------------------------

            (
                annotated_frame,
                keypoints,
                landmark_confidence,
                person_confidence
            ) = (
                self.pose_detector.process_frame(
                    frame
                )
            )

            if keypoints is None:

                self.activity_label.configure(
                    text="NO PERSON"
                )

                self.confidence_label.configure(
                    text="Confidence: --"
                )

                self.exercise_label.configure(
                    text="Exercise: NONE"
                )

                self.rep_label.configure(
                    text="0"
                )

                self.feedback_label.configure(
                    text=(
                        "No person detected in the image."
                    )
                )

                self.display_image(
                    annotated_frame
                )

                return

            # --------------------------------------------------
            # ACTIVITY
            # --------------------------------------------------

            (
                activity,
                confidence
            ) = (
                self.activity_detector.detect(
                    keypoints,
                    landmark_confidence
                )
            )

            self.activity_label.configure(
                text=activity
            )

            self.confidence_label.configure(
                text=(
                    f"Confidence: "
                    f"{confidence * 100:.0f}%"
                )
            )

            # --------------------------------------------------
            # IMAGE DOES NOT CONTAIN TEMPORAL MOVEMENT
            # --------------------------------------------------

            self.exercise_label.configure(
                text="Exercise: IMAGE ANALYSIS"
            )

            self.rep_label.configure(
                text="--"
            )

            # --------------------------------------------------
            # Exercise posture analysis
            # --------------------------------------------------

            if activity == (
                ActivityDetector.BODY_NOT_VISIBLE
            ):

                feedback = (
                    "Full body is not visible. "
                    "Upload a full-body image."
                )

            elif activity == (
                ActivityDetector.LYING
            ):

                feedback = (
                    "Person appears to be lying down."
                )

            elif activity == (
                ActivityDetector.SITTING
            ):

                feedback = (
                    "Person appears to be sitting."
                )

            else:

                feedback = (
                    "Image analyzed successfully."
                )

            self.feedback_label.configure(
                text=feedback
            )

            self.status_label.configure(
                text="Image analysis completed"
            )

            # --------------------------------------------------
            # DISPLAY RESULT
            # --------------------------------------------------

            self.display_image(
                annotated_frame
            )

        except Exception as error:

            messagebox.showerror(
                "Analysis Error",
                str(error)
            )

    # ======================================================
    # DISPLAY IMAGE
    # ======================================================

    def display_image(
        self,
        frame
    ):

        try:

            rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            image = Image.fromarray(
                rgb
            )

            width = max(
                self.video_frame.winfo_width() - 20,
                640
            )

            height = max(
                self.video_frame.winfo_height() - 20,
                400
            )

            image.thumbnail(
                (width, height),
                Image.Resampling.LANCZOS
            )

            ctk_image = ctk.CTkImage(
                light_image=image,
                dark_image=image,
                size=image.size
            )

            self.video_label.configure(
                image=ctk_image,
                text=""
            )

            self.video_label.image = ctk_image

        except Exception as error:

            print(
                f"Image display error: {error}"
            )

    # ======================================================
    # CAMERA FRAME LOOP
    # ======================================================

    def update_frame(self):

        if not self.camera_running:
            return

        try:

            if self.camera is None:
                return

            success, frame = (
                self.camera.read()
            )

            if not success:

                self.status_label.configure(
                    text="Unable to read camera frame"
                )

                self.root.after(
                    100,
                    self.update_frame
                )

                return

            # Mirror webcam.
            frame = cv2.flip(
                frame,
                1
            )

            # --------------------------------------------------
            # YOLO POSE
            # --------------------------------------------------

            (
                annotated_frame,
                keypoints,
                landmark_confidence,
                person_confidence
            ) = (
                self.pose_detector.process_frame(
                    frame
                )
            )

            # --------------------------------------------------
            # NO PERSON
            # --------------------------------------------------

            if keypoints is None:

                self.activity_label.configure(
                    text="NO PERSON"
                )

                self.confidence_label.configure(
                    text="Confidence: --"
                )

                self.exercise_label.configure(
                    text="Exercise: NONE"
                )

                self.feedback_label.configure(
                    text=(
                        "Move into the camera frame."
                    )
                )

            else:

                # --------------------------------------------------
                # ACTIVITY
                # --------------------------------------------------

                (
                    activity,
                    activity_confidence
                ) = (
                    self.activity_detector.detect(
                        keypoints,
                        landmark_confidence
                    )
                )

                self.activity_label.configure(
                    text=activity
                )

                self.confidence_label.configure(
                    text=(
                        f"Confidence: "
                        f"{activity_confidence * 100:.0f}%"
                    )
                )

                # --------------------------------------------------
                # BODY VISIBILITY
                # --------------------------------------------------

                if activity == (
                    ActivityDetector.BODY_NOT_VISIBLE
                ):

                    self.exercise_label.configure(
                        text="Exercise: NONE"
                    )

                    self.rep_label.configure(
                        text="0"
                    )

                    self.feedback_label.configure(
                        text=(
                            "Move farther away and "
                            "show your complete body."
                        )
                    )

                else:

                    # --------------------------------------------------
                    # EXERCISE
                    # --------------------------------------------------

                    (
                        reps,
                        exercise,
                        feedback
                    ) = (
                        self.exercise_tracker.process(
                            keypoints,
                            landmark_confidence
                        )
                    )

                    self.rep_label.configure(
                        text=str(reps)
                    )

                    self.exercise_label.configure(
                        text=(
                            f"Exercise: {exercise}"
                        )
                    )

                    self.feedback_label.configure(
                        text=feedback
                    )

            # --------------------------------------------------
            # FPS
            # --------------------------------------------------

            self.total_frames += 1

            now = time.time()

            elapsed = (
                now -
                self.last_fps_time
            )

            if elapsed >= 1.0:

                self.fps = (
                    self.total_frames /
                    elapsed
                )

                self.total_frames = 0

                self.last_fps_time = now

                self.fps_label.configure(
                    text=(
                        f"FPS: {self.fps:.1f}"
                    )
                )

            # --------------------------------------------------
            # DISPLAY
            # --------------------------------------------------

            self.display_image(
                annotated_frame
            )

        except Exception as error:

            print(
                f"Frame update error: {error}"
            )

        if self.camera_running:

            self.root.after(
                15,
                self.update_frame
            )

    # ======================================================
    # CLOSE APPLICATION
    # ======================================================

    def close_application(self):

        self.camera_running = False

        try:

            if self.camera is not None:

                self.camera.release()

                self.camera = None

        except Exception:
            pass

        try:

            cv2.destroyAllWindows()

        except Exception:
            pass

        self.root.destroy()


# ==========================================================
# START APPLICATION
# ==========================================================

if __name__ == "__main__":

    root = ctk.CTk()

    app = HARApplication(
        root
    )

    root.mainloop()