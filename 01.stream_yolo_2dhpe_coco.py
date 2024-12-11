import cv2
import os 
from ultralytics import YOLO
import numpy as np
from utils import drawkeypoints

# YOLO Pose 모델 로드
model = YOLO("yolo11n-pose.pt")  # 원하는 모델을 다운로드 후 경로 수정 가능

# 웹캠 캡처 초기화
cap = cv2.VideoCapture(0)  # 0은 기본 웹캠

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

output_folder = "frames"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

frame_cnt = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # YOLO 모델로 프레임 추론
    results = model(frame, verbose=False)

    # 결과에서 keypoints 가져오기
    for result in results:
        keypoints = result.keypoints.data.cpu().numpy()  # numpy 배열로 변환
        frame = drawkeypoints(frame, keypoints)  # 스켈레톤 그리기

    frame_filename = os.path.join(output_folder, f"{frame_cnt:03d}.jpg")
    cv2.imwrite(frame_filename, frame)
    frame_cnt += 1
    
    # 프레임 출력
    cv2.imshow('Pose Estimation', frame)

    # 'q'를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
