// ============================================
// AI HUMAN ACTIVITY RECOGNITION
// Frontend Controller - FIXED VERSION
// ============================================

const camera = document.getElementById("camera");
const overlayCanvas = document.getElementById("overlayCanvas");
const uploadedImage = document.getElementById("uploadedImage");
const cameraPlaceholder = document.getElementById("cameraPlaceholder");

const startCameraBtn = document.getElementById("startCameraBtn");
const stopCameraBtn = document.getElementById("stopCameraBtn");
const imageInput = document.getElementById("imageInput");
const exerciseSelect = document.getElementById("exerciseSelect");

const systemStatus = document.getElementById("systemStatus");
const videoStatus = document.getElementById("videoStatus");
const liveBadge = document.getElementById("liveBadge");

const activityValue = document.getElementById("activityValue");
const confidenceValue = document.getElementById("confidenceValue");
const repValue = document.getElementById("repValue");
const angleValue = document.getElementById("angleValue");
const exerciseValue = document.getElementById("exerciseValue");
const personValue = document.getElementById("personValue");

const feedbackMessage = document.getElementById("feedbackMessage");

const uploadResult = document.getElementById("uploadResult");
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
// CHECK ELEMENTS
// ============================================

console.log("HAR frontend loaded");

if (!activityValue) {
    console.error("Missing #activityValue");
}

if (!confidenceValue) {
    console.error("Missing #confidenceValue");
}

if (!personValue) {
    console.error("Missing #personValue");
}

if (!imageInput) {
    console.error("Missing #imageInput");
}


// ============================================
// EXERCISE SELECTOR
// ============================================

if (exerciseSelect) {

    exerciseSelect.addEventListener(
        "change",
        function () {

            const exercise =
                exerciseNames[this.value] ||
                formatExerciseName(this.value);

            if (exerciseValue) {
                exerciseValue.textContent = exercise;
            }

            if (repValue) {
                repValue.textContent = "0";
            }

            if (angleValue) {
                angleValue.textContent = "--";
            }

            showFeedback(
                `${exercise} selected. Ready to analyze.`,
                "ready"
            );
        }
    );
}


// ============================================
// START CAMERA
// ============================================

if (startCameraBtn) {
    startCameraBtn.addEventListener(
        "click",
        startCamera
    );
}


async function startCamera() {

    try {

        if (
            !navigator.mediaDevices ||
            !navigator.mediaDevices.getUserMedia
        ) {

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


        if (camera) {
            camera.srcObject = cameraStream;
            camera.style.display = "block";
        }

        if (uploadedImage) {
            uploadedImage.style.display = "none";
        }

        if (cameraPlaceholder) {
            cameraPlaceholder.style.display = "none";
        }

        if (liveBadge) {
            liveBadge.style.display = "block";
        }

        if (systemStatus) {
            systemStatus.textContent = "Camera Active";
        }

        if (videoStatus) {
            videoStatus.textContent =
                "Live camera analysis running";
        }

        cameraRunning = true;

        showFeedback(
            "Camera started. Position your full body inside the frame.",
            "ready"
        );

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

if (stopCameraBtn) {
    stopCameraBtn.addEventListener(
        "click",
        stopCamera
    );
}


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


    if (camera) {
        camera.srcObject = null;
        camera.style.display = "none";
    }

    if (liveBadge) {
        liveBadge.style.display = "none";
    }

    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "flex";
    }

    if (systemStatus) {
        systemStatus.textContent = "System Ready";
    }

    if (videoStatus) {
        videoStatus.textContent =
            "Camera is currently stopped";
    }

    if (activityValue) {
        activityValue.textContent = "WAITING";
    }

    if (confidenceValue) {
        confidenceValue.textContent = "--";
    }

    if (personValue) {
        personValue.textContent = "Not Detected";
    }

    showFeedback(
        "Camera stopped.",
        "ready"
    );
}


// ============================================
// IMAGE UPLOAD
// ============================================

if (imageInput) {

    imageInput.addEventListener(
        "change",
        handleImageUpload
    );
}


async function handleImageUpload(event) {

    const file =
        event.target.files[0];


    if (!file) {
        return;
    }


    console.log("================================");
    console.log("IMAGE UPLOAD STARTED");
    console.log("File:", file.name);
    console.log("Type:", file.type);
    console.log("Size:", file.size);
    console.log("================================");


    // Stop webcam if running
    if (cameraRunning) {
        stopCamera();
    }


    const imageURL =
        URL.createObjectURL(file);


    if (uploadedImage) {
        uploadedImage.src = imageURL;
        uploadedImage.style.display = "block";
    }

    if (camera) {
        camera.style.display = "none";
    }

    if (cameraPlaceholder) {
        cameraPlaceholder.style.display = "none";
    }


    if (uploadResult) {
        uploadResult.style.display = "block";
    }


    if (uploadResultContent) {
        uploadResultContent.textContent =
            "Analyzing image...";
    }


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
        exerciseSelect
            ? exerciseSelect.value
            : "squat"
    );


    // ========================================
    // API REQUEST
    // ========================================

    let response;

    try {

        console.log(
            "Sending POST /analyze-image"
        );

        response =
            await fetch(
                "/analyze-image",
                {
                    method: "POST",
                    body: formData
                }
            );

    }

    catch (error) {

        console.error(
            "NETWORK ERROR:",
            error
        );

        if (uploadResultContent) {
            uploadResultContent.textContent =
                "Unable to connect to the analysis server.";
        }

        showFeedback(
            "Could not connect to the AI server.",
            "error"
        );

        return;
    }


    // ========================================
    // CHECK HTTP RESPONSE
    // ========================================

    console.log(
        "Server status:",
        response.status
    );


    let rawResponse = "";

    try {

        rawResponse =
            await response.text();

        console.log(
            "Raw server response:",
            rawResponse
        );

    }

    catch (error) {

        console.error(
            "Could not read server response:",
            error
        );

        if (uploadResultContent) {
            uploadResultContent.textContent =
                "Could not read server response.";
        }

        showFeedback(
            "The server response could not be read.",
            "error"
        );

        return;
    }


    if (!response.ok) {

        console.error(
            "SERVER ERROR:",
            response.status,
            rawResponse
        );

        if (uploadResultContent) {
            uploadResultContent.textContent =
                `Server error (${response.status})`;
        }

        showFeedback(
            `AI server returned error ${response.status}.`,
            "error"
        );

        return;
    }


    // ========================================
    // PARSE JSON
    // ========================================

    let result;

    try {

        result =
            JSON.parse(rawResponse);

    }

    catch (error) {

        console.error(
            "JSON PARSE ERROR:",
            error
        );

        console.error(
            "Server returned:",
            rawResponse
        );

        if (uploadResultContent) {
            uploadResultContent.textContent =
                "The AI server returned an invalid response.";
        }

        showFeedback(
            "The server responded, but the response format was invalid.",
            "error"
        );

        return;
    }


    // ========================================
    // SUCCESS
    // ========================================

    console.log(
        "AI RESULT:",
        result
    );


    // Update dashboard separately
    try {

        updateAnalysisUI(result);

    }

    catch (error) {

        console.error(
            "UI UPDATE ERROR:",
            error
        );

        showFeedback(
            "AI analysis completed, but the dashboard could not be updated.",
            "warning"
        );
    }


    // ========================================
    // DISPLAY IMAGE ANALYSIS RESULT
    // ========================================

    if (uploadResultContent) {

        uploadResultContent.innerHTML = `

            <div>
                <strong>Activity:</strong>
                ${escapeHTML(
                    getActivityName(result)
                )}
            </div>

            <div>
                <strong>Exercise:</strong>
                ${escapeHTML(
                    result.exercise ||
                    "Unknown"
                )}
            </div>

            <div>
                <strong>Confidence:</strong>
                ${formatConfidence(
                    result.activity_confidence ??
                    result.confidence
                )}
            </div>

            <div>
                <strong>Angle:</strong>
                ${
                    result.angle !== null &&
                    result.angle !== undefined
                        ? Number(result.angle).toFixed(1)
                        : "--"
                }°
            </div>

            <div>
                <strong>Person:</strong>
                ${
                    result.person_detected
                        ? "Detected"
                        : "Not Detected"
                }
            </div>

            <div>
                <strong>Feedback:</strong>
                ${escapeHTML(
                    result.feedback ||
                    "No feedback available."
                )}
            </div>

        `;
    }


    if (result.feedback) {

        const feedback =
            String(result.feedback)
                .toLowerCase();

        let feedbackClass = "ready";


        if (
            feedback.includes("good") ||
            feedback.includes("correct") ||
            feedback.includes("perfect") ||
            feedback.includes("excellent")
        ) {

            feedbackClass = "good";
        }

        else if (
            feedback.includes("lower") ||
            feedback.includes("raise") ||
            feedback.includes("adjust") ||
            feedback.includes("warning")
        ) {

            feedbackClass = "warning";
        }

        else if (
            feedback.includes("not") ||
            feedback.includes("error") ||
            feedback.includes("unable")
        ) {

            feedbackClass = "error";
        }


        showFeedback(
            result.feedback,
            feedbackClass
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


    analysisTimer =
        setInterval(
            captureAndAnalyzeFrame,
            500
        );
}


async function captureAndAnalyzeFrame() {

    if (
        !cameraRunning ||
        !camera ||
        camera.readyState < 2
    ) {

        return;
    }


    if (
        !camera.videoWidth ||
        !camera.videoHeight
    ) {

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


    canvas.width = width;
    canvas.height = height;


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
                exerciseSelect
                    ? exerciseSelect.value
                    : "squat"
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

                    console.error(
                        "Frame API error:",
                        response.status
                    );

                    return;
                }


                const result =
                    await response.json();


                updateAnalysisUI(result);

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


    console.log(
        "Updating UI with:",
        result
    );


    // ========================================
    // ACTIVITY
    // ========================================

    if (activityValue) {

        activityValue.textContent =
            getActivityName(result)
                .toUpperCase();
    }


    // ========================================
    // CONFIDENCE
    // ========================================

    if (confidenceValue) {

        confidenceValue.textContent =
            formatConfidence(
                result.activity_confidence ??
                result.confidence
            );
    }


    // ========================================
    // REPETITIONS
    // ========================================

    if (repValue) {

        repValue.textContent =
            result.repetitions ??
            result.reps ??
            0;
    }


    // ========================================
    // ANGLE
    // ========================================

    if (angleValue) {

        if (
            result.angle !== null &&
            result.angle !== undefined &&
            !Number.isNaN(
                Number(result.angle)
            )
        ) {

            angleValue.textContent =
                Number(result.angle)
                    .toFixed(1);

        }

        else {

            angleValue.textContent =
                "--";
        }
    }


    // ========================================
    // EXERCISE
    // ========================================

    if (
        exerciseValue &&
        result.exercise
    ) {

        exerciseValue.textContent =
            formatExerciseName(
                result.exercise
            );
    }


    // ========================================
    // PERSON
    // ========================================

    if (personValue) {

        if (
            result.person_detected === true
        ) {

            personValue.textContent =
                "Detected";
        }

        else {

            personValue.textContent =
                "Not Detected";
        }
    }


    // ========================================
    // FEEDBACK
    // ========================================

    if (result.feedback) {

        let feedbackClass =
            "ready";


        const feedback =
            String(result.feedback)
                .toLowerCase();


        if (
            feedback.includes("good") ||
            feedback.includes("correct") ||
            feedback.includes("perfect") ||
            feedback.includes("excellent")
        ) {

            feedbackClass = "good";
        }

        else if (
            feedback.includes("lower") ||
            feedback.includes("raise") ||
            feedback.includes("adjust") ||
            feedback.includes("warning")
        ) {

            feedbackClass = "warning";
        }

        else if (
            feedback.includes("not") ||
            feedback.includes("error") ||
            feedback.includes("unable")
        ) {

            feedbackClass = "error";
        }


        showFeedback(
            result.feedback,
            feedbackClass
        );
    }

}


// ============================================
// ACTIVITY NAME
// ============================================

function getActivityName(result) {

    if (!result) {
        return "Unknown";
    }


    const activity =
        result.activity ||
        result.predicted_activity ||
        result.detected_activity ||
        result.label ||
        "Unknown";


    return normalizeActivity(
        String(activity)
    );
}


// ============================================
// NORMALIZE ACTIVITY
// ============================================

function normalizeActivity(activity) {

    const value =
        activity
            .trim()
            .toLowerCase()
            .replaceAll("_", " ");


    if (
        value === "squat" ||
        value === "squatting"
    ) {

        return "Squatting";
    }


    if (
        value === "sit" ||
        value === "sitting" ||
        value === "seated"
    ) {

        return "Sitting";
    }


    if (
        value === "stand" ||
        value === "standing"
    ) {

        return "Standing";
    }


    if (
        value === "lie" ||
        value === "lying" ||
        value === "lying down" ||
        value === "laying"
    ) {

        return "Lying Down";
    }


    if (
        value === "walking" ||
        value === "walk"
    ) {

        return "Walking";
    }


    if (
        value === "unknown" ||
        value === "none"
    ) {

        return "Unknown";
    }


    return activity;
}


// ============================================
// FEEDBACK
// ============================================

function showFeedback(
    message,
    type = "ready"
) {

    if (!feedbackMessage) {
        return;
    }


    feedbackMessage.textContent =
        message;


    feedbackMessage.className =
        `feedback-message ${type}`;
}


// ============================================
// FORMAT CONFIDENCE
// ============================================

function formatConfidence(
    confidence
) {

    if (
        confidence === undefined ||
        confidence === null ||
        confidence === ""
    ) {

        return "--";
    }


    const value =
        Number(confidence);


    if (Number.isNaN(value)) {
        return "--";
    }


    const percentage =
        value <= 1
            ? value * 100
            : value;


    return `${percentage.toFixed(1)}%`;
}


// ============================================
// FORMAT EXERCISE NAME
// ============================================

function formatExerciseName(
    exercise
) {

    if (!exercise) {
        return "Unknown";
    }


    return String(exercise)
        .replaceAll("_", " ")
        .replace(
            /\b\w/g,
            letter =>
                letter.toUpperCase()
        );
}


// ============================================
// ESCAPE HTML
// ============================================

function escapeHTML(value) {

    return String(value)
        .replaceAll(
            "&",
            "&amp;"
        )
        .replaceAll(
            "<",
            "&lt;"
        )
        .replaceAll(
            ">",
            "&gt;"
        )
        .replaceAll(
            '"',
            "&quot;"
        )
        .replaceAll(
            "'",
            "&#039;"
        );
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


// ============================================
// GLOBAL ERROR LOGGING
// ============================================

window.addEventListener(
    "error",
    function (event) {

        console.error(
            "GLOBAL JAVASCRIPT ERROR:",
            event.error || event.message
        );
    }
);


window.addEventListener(
    "unhandledrejection",
    function (event) {

        console.error(
            "UNHANDLED PROMISE ERROR:",
            event.reason
        );
    }
);