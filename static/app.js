// ============================================
// AI HUMAN ACTIVITY RECOGNITION
// Frontend Controller
// ============================================


const camera = document.getElementById("camera");

const overlayCanvas =
    document.getElementById("overlayCanvas");

const uploadedImage =
    document.getElementById("uploadedImage");

const cameraPlaceholder =
    document.getElementById("cameraPlaceholder");

const startCameraBtn =
    document.getElementById("startCameraBtn");

const stopCameraBtn =
    document.getElementById("stopCameraBtn");

const imageInput =
    document.getElementById("imageInput");

const exerciseSelect =
    document.getElementById("exerciseSelect");

const systemStatus =
    document.getElementById("systemStatus");

const videoStatus =
    document.getElementById("videoStatus");

const liveBadge =
    document.getElementById("liveBadge");

const activityValue =
    document.getElementById("activityValue");

const confidenceValue =
    document.getElementById("confidenceValue");

const repValue =
    document.getElementById("repValue");

const angleValue =
    document.getElementById("angleValue");

const exerciseValue =
    document.getElementById("exerciseValue");

const personValue =
    document.getElementById("personValue");

const feedbackMessage =
    document.getElementById("feedbackMessage");

const uploadResult =
    document.getElementById("uploadResult");

const uploadResultContent =
    document.getElementById("uploadResultContent");


let cameraStream = null;

let cameraRunning = false;

let analysisTimer = null;


// ============================================
// EXERCISE NAMES
// ============================================

const exerciseNames = {

    squat: "Squat",

    bicep_curl: "Bicep Curl",

    shoulder_press: "Shoulder Press"

};


// ============================================
// EXERCISE SELECTOR
// ============================================

exerciseSelect.addEventListener(
    "change",
    function () {

        const exercise =
            exerciseNames[this.value];

        exerciseValue.textContent =
            exercise;

        // Reset UI counters
        repValue.textContent = "0";

        angleValue.textContent = "--";

        feedbackMessage.textContent =
            `${exercise} selected. Ready to analyze.`;

        feedbackMessage.className =
            "feedback-message ready";

    }
);


// ============================================
// START CAMERA
// ============================================

startCameraBtn.addEventListener(
    "click",
    startCamera
);


async function startCamera() {

    try {

        if (!navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia) {

            showFeedback(
                "Camera access is not supported by this browser.",
                "error"
            );

            return;
        }


        cameraStream =
            await navigator.mediaDevices.getUserMedia({

                video: {
                    width: {
                        ideal: 640
                    },

                    height: {
                        ideal: 480
                    },

                    facingMode: "user"
                },

                audio: false

            });


        camera.srcObject =
            cameraStream;

        camera.style.display =
            "block";

        uploadedImage.style.display =
            "none";

        cameraPlaceholder.style.display =
            "none";

        liveBadge.style.display =
            "block";

        systemStatus.textContent =
            "Camera Active";

        videoStatus.textContent =
            "Live camera analysis running";

        cameraRunning = true;


        showFeedback(
            "Camera started. Position your full body inside the frame.",
            "ready"
        );


        // Start sending frames
        startFrameAnalysis();

    }

    catch (error) {

        console.error(
            "Camera error:",
            error
        );

        showFeedback(
            "Unable to access camera. Please allow camera permission.",
            "error"
        );

    }

}


// ============================================
// STOP CAMERA
// ============================================

stopCameraBtn.addEventListener(
    "click",
    stopCamera
);


function stopCamera() {

    cameraRunning = false;


    if (analysisTimer) {

        clearInterval(
            analysisTimer
        );

        analysisTimer = null;

    }


    if (cameraStream) {

        cameraStream
            .getTracks()
            .forEach(
                track => track.stop()
            );

        cameraStream = null;

    }


    camera.srcObject = null;

    camera.style.display =
        "none";

    liveBadge.style.display =
        "none";

    cameraPlaceholder.style.display =
        "flex";

    systemStatus.textContent =
        "System Ready";

    videoStatus.textContent =
        "Camera is currently stopped";


    activityValue.textContent =
        "WAITING";

    confidenceValue.textContent =
        "--";

    personValue.textContent =
        "Not Detected";


    showFeedback(
        "Camera stopped.",
        "ready"
    );

}


// ============================================
// IMAGE UPLOAD
// ============================================

imageInput.addEventListener(
    "change",
    handleImageUpload
);


async function handleImageUpload(event) {

    const file =
        event.target.files[0];


    if (!file) {
        return;
    }


    // Stop webcam if running
    if (cameraRunning) {
        stopCamera();
    }


    const imageURL =
        URL.createObjectURL(file);


    uploadedImage.src =
        imageURL;

    uploadedImage.style.display =
        "block";

    camera.style.display =
        "none";

    cameraPlaceholder.style.display =
        "none";


    uploadResult.style.display =
        "block";


    uploadResultContent.textContent =
        "Analyzing image...";


    showFeedback(
        "Image uploaded. AI analysis is starting...",
        "ready"
    );


    const formData =
        new FormData();

    formData.append(
        "file",
        file
    );

    formData.append(
        "exercise",
        exerciseSelect.value
    );


    try {

        const response =
            await fetch(
                "/analyze-image",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                `Server returned ${response.status}`
            );

        }


        const result =
            await response.json();


        updateAnalysisUI(
            result
        );


        uploadResultContent.innerHTML = `

            <div>
                <strong>Activity:</strong>
                ${escapeHTML(result.activity || "Unknown")}
            </div>

            <div>
                <strong>Exercise:</strong>
                ${escapeHTML(result.exercise || "Unknown")}
            </div>

            <div>
                <strong>Confidence:</strong>
                ${formatConfidence(result.confidence)}
            </div>

            <div>
                <strong>Angle:</strong>
                ${result.angle ?? "--"}°
            </div>

            <div>
                <strong>Feedback:</strong>
                ${escapeHTML(result.feedback || "No feedback")}
            </div>

        `;

    }

    catch (error) {

        console.error(
            "Upload analysis error:",
            error
        );


        uploadResultContent.textContent =
            "Image analysis failed.";


        showFeedback(
            "The server could not analyze this image yet.",
            "error"
        );

    }

}


// ============================================
// LIVE FRAME ANALYSIS
// ============================================

function startFrameAnalysis() {

    if (analysisTimer) {

        clearInterval(
            analysisTimer
        );

    }


    // Send a frame approximately every 300 ms
    analysisTimer =
        setInterval(
            captureAndAnalyzeFrame,
            300
        );

}


async function captureAndAnalyzeFrame() {

    if (!cameraRunning ||
        camera.readyState < 2) {

        return;

    }


    const canvas =
        document.createElement("canvas");


    const width = 640;

    const height =
        Math.round(
            camera.videoHeight *
            (width / camera.videoWidth)
        );


    canvas.width =
        width;

    canvas.height =
        height;


    const context =
        canvas.getContext("2d");


    context.drawImage(
        camera,
        0,
        0,
        width,
        height
    );


    canvas.toBlob(
        async function (blob) {

            if (!blob) {
                return;
            }


            const formData =
                new FormData();


            formData.append(
                "file",
                blob,
                "camera.jpg"
            );


            formData.append(
                "exercise",
                exerciseSelect.value
            );


            try {

                const response =
                    await fetch(
                        "/analyze-frame",
                        {
                            method: "POST",
                            body: formData
                        }
                    );


                if (!response.ok) {
                    return;
                }


                const result =
                    await response.json();


                updateAnalysisUI(
                    result
                );


            }

            catch (error) {

                console.error(
                    "Frame analysis error:",
                    error
                );

            }

        },
        "image/jpeg",
        0.7
    );

}


// ============================================
// UPDATE UI
// ============================================

function updateAnalysisUI(result) {

    if (!result) {
        return;
    }


    activityValue.textContent =
        (result.activity ||
            "UNKNOWN").toUpperCase();


    confidenceValue.textContent =
        formatConfidence(
            result.confidence
        );


    repValue.textContent =
        result.repetitions ??
        result.reps ??
        0;


    angleValue.textContent =
        result.angle ??
        "--";


    if (result.exercise) {

        exerciseValue.textContent =
            formatExerciseName(
                result.exercise
            );

    }


    if (result.person_detected) {

        personValue.textContent =
            "Detected";

    }

    else if (
        result.person_detected === false
    ) {

        personValue.textContent =
            "Not Detected";

    }


    if (result.feedback) {

        let feedbackClass =
            "ready";


        const feedback =
            result.feedback.toLowerCase();


        if (
            feedback.includes("good") ||
            feedback.includes("correct") ||
            feedback.includes("perfect")
        ) {

            feedbackClass =
                "good";

        }

        else if (
            feedback.includes("lower") ||
            feedback.includes("raise") ||
            feedback.includes("adjust")
        ) {

            feedbackClass =
                "warning";

        }

        else if (
            feedback.includes("not") ||
            feedback.includes("error")
        ) {

            feedbackClass =
                "error";

        }


        showFeedback(
            result.feedback,
            feedbackClass
        );

    }

}


// ============================================
// FEEDBACK
// ============================================

function showFeedback(
    message,
    type = "ready"
) {

    feedbackMessage.textContent =
        message;

    feedbackMessage.className =
        `feedback-message ${type}`;

}


// ============================================
// HELPERS
// ============================================

function formatConfidence(
    confidence
) {

    if (
        confidence === undefined ||
        confidence === null
    ) {

        return "--";

    }


    let value =
        Number(confidence);


    if (Number.isNaN(value)) {

        return "--";

    }


    if (value <= 1) {

        value *= 100;

    }


    return `${value.toFixed(1)}%`;

}


function formatExerciseName(
    exercise
) {

    return exercise
        .replaceAll("_", " ")
        .replace(/\b\w/g,
            letter => letter.toUpperCase()
        );

}


function escapeHTML(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

}


// ============================================
// PAGE CLEANUP
// ============================================

window.addEventListener(
    "beforeunload",
    function () {

        if (cameraStream) {

            cameraStream
                .getTracks()
                .forEach(
                    track => track.stop()
                );

        }

    }
);