import cv2
from ultralytics import YOLO
import numpy as np
from utils import h36m_coco_format, show2Dpose
import os 
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
    
    frame = cv2.flip(frame,1)
    # YOLO 모델로 프레임 추론
    try:
        results = model(frame, verbose=False)

        # 결과에서 keypoints 가져오기
        for result in results:
            jointsdata = result.keypoints.data.cpu().numpy() 
            keypoints, scores = jointsdata[:1,:,:2], jointsdata[:1,:,2:]
            if scores.any():
                h36m_kpts, h36m_scores, valid_frames = h36m_coco_format(keypoints, scores)
                frame = show2Dpose(h36m_kpts, frame)

    except:
        print('something error')
    
    
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
