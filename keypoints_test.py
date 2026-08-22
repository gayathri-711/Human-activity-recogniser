from ultralytics import YOLO
import cv2

model = YOLO("yolo11n-pose.pt")

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()

    if not success:
        break

    results = model(frame, verbose=False)

    if len(results) > 0:

        keypoints = results[0].keypoints

        if keypoints is not None:

            xy = keypoints.xy.cpu().numpy()

            if len(xy) > 0:

                points = xy[0]

                print("\n========== BODY POINTS ==========")

                for i, point in enumerate(points):
                    x, y = point
                    print(f"{i}: ({int(x)}, {int(y)})")

    annotated = results[0].plot()

    cv2.imshow("Keypoint Test", annotated)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
