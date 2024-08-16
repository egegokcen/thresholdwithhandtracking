import cv2
import mediapipe as mp

def nothing(x):
    pass

def thresholdSetter(x):
    threshold = int(255 * x / 1280)
    cv2.setTrackbarPos("Threshold", "Frame", threshold)
    return threshold

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
fingerCoordinates = [(8, 6), (12, 10), (16, 14), (20, 18)]
thumbCoordinates = (4, 2)

cv2.namedWindow("Frame")
cv2.createTrackbar("Threshold", "Frame", 0, 255, nothing)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1024)

threshold = 10

while True:
    success, frame = cap.read()
    if not success:
        print("Cam is not captured")
        break
    
    frame = cv2.flip(frame, 1)
    frameRgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frameRgb)
    multiLandMarks = results.multi_hand_landmarks

    if multiLandMarks:
        handPoints = []
        for handLms in multiLandMarks:
            mpDraw.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)
            for lm in handLms.landmark:
                h, w, _ = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                handPoints.append((cx, cy))

        upCount = 0
        for coordinate in fingerCoordinates:
            if handPoints[coordinate[0]][1] < handPoints[coordinate[1]][1]:
                upCount += 1

        if handPoints[thumbCoordinates[0]][0] < handPoints[thumbCoordinates[1]][0]:
            upCount += 1

        cv2.putText(frame, str(upCount), (150, 150), cv2.FONT_HERSHEY_PLAIN, 12, (255, 0, 0), 12)

        if upCount > 0:
            value = thresholdSetter(handPoints[coordinate[1]][0])
            threshold = value
            
        # else:
        #     cv2.imshow("Frame", frame)

        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, threshFrame = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY)
        threshFrame = cv2.cvtColor(threshFrame, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Frame", threshFrame)


    else:
        grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, threshFrame = cv2.threshold(grayFrame, threshold, 255, cv2.THRESH_BINARY)
        threshFrame = cv2.cvtColor(threshFrame, cv2.COLOR_GRAY2BGR)
        cv2.imshow("Frame", threshFrame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        print("Leaving")
        break

cap.release()
cv2.destroyAllWindows()
