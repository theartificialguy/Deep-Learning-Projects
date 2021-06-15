import cv2
import time
import mediapipe as mp

cap = cv2.VideoCapture('video.mp4')

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

prevTime = 0

while cap.isOpened():

    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(frame, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
       
    currTime = time.time()
    fps = 1/(currTime - prevTime)
    prevTime = currTime
    
    cv2.putText(frame, str(int(fps)), (70, 50), cv2.FONT_HERSHEY_SIMPLEX, 3, (0,0,255), 2)
    cv2.imshow("output", frame)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()