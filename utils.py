import numpy as np
import cv2



# 색상 팔레트
COLORS = {
    "joint": (0, 255, 0),  # 초록색 (관절점)
    "line": (255, 0, 0)    # 파란색 (관절 연결선)
}

def drawkeypoints(frame, keypoints, color=(0, 255, 0), thickness=5):
    for keypoint in keypoints:
        # 신뢰도가 50% 이상인 포인트만 처리
        points = [(int(x), int(y)) if conf > 0.6 else None for x, y, conf in keypoint]
        # points = [(int(x), int(y)) for x, y, conf in keypoint]
        
        # 관절 포인트를 그림
        for i, point in enumerate(points):
            # if i == 0:
            #     color = (255, 0, 0)
            # else:
            #     color = (0, 255, 0)
            if point:  # 유효한 점일 때만 그림
                cv2.circle(frame, point, thickness, color, -1)
                
    return frame

def drawkeypoints2(frame, keypoints, color=(0, 255, 0), thickness=5):
    for keypoint in keypoints:
        # 신뢰도가 50% 이상인 포인트만 처리
        points = [(int(x), int(y)) if conf > 0.6 else None for x, y, conf in keypoint]
        # points = [(int(x), int(y)) for x, y, conf in keypoint]
        
        # 관절 포인트를 그림
        for i, point in enumerate(points):
            if i == 7:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)
            if point:  # 유효한 점일 때만 그림
                cv2.circle(frame, point, thickness, color, -1)
                
    return frame

def show2Dpose_h36m(frame, keypoints, colors=COLORS):

    for keypoint in keypoints:
        # 신뢰도가 50% 이상인 포인트만 처리
        points = [(int(x), int(y)) if conf > 0.5 else None for x, y, conf in keypoint]
        
        # 관절 포인트를 그림
        for point in points:
            if point:  # 유효한 점일 때만 그림
                cv2.circle(frame, point, 5, colors["joint"], -1)
                
    return frame


def show2Dpose(kps, img):
    kps = kps.reshape(17,2)
    connections = [[0, 1], [1, 2], [2, 3], 
                   [0, 4], [4, 5], [5, 6], 
                   [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], 
                   [8, 14], [14, 15], [15, 16]]

    LR = [0, 0, 0, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 0, 0, 0]

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    ccolor = (0, 150, 0)
    thickness = 3

    for j, c in enumerate(connections):
        start_idx, end_idx = c
        start = list(map(int, kps[start_idx]))
        end = list(map(int, kps[end_idx]))
        
        # Skip visualization if either keypoint is (0, 0)
        if (start[0] == 0 and start[1] == 0) or (end[0] == 0 and end[1] == 0):
            continue
        
        # Draw the line between keypoints
        cv2.line(img, (start[0], start[1]), (end[0], end[1]),lcolor if LR[j] == 1 else (rcolor if LR[j] == 0 else ccolor), thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]

import numpy as np

h36m_coco_order = [9, 11, 14, 12, 15, 13, 16, 4, 1, 5, 2, 6, 3]
coco_order = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
spple_keypoints = [10, 8, 0, 7]

def coco_h36m(keypoints):
    temporal = keypoints.shape[0]
    keypoints_h36m = np.zeros_like(keypoints, dtype=np.float32)
    htps_keypoints = np.zeros((temporal, 4, 2), dtype=np.float32)

    # htps_keypoints: head(0:10), thorax(1:8), pelvis(2:0), spine(3:7)
    # Head (10의 x 값)
    valid_points = [k for k in keypoints[:, 1:5, 0][0] if k != 0]
    htps_keypoints[:, 0, 0] = np.mean(valid_points) if valid_points else 0.0

    # Head (10의 y 값)
    valid_points_y = [k for k in keypoints[:, 1:3, 1][0] if k != 0]
    mean_y = np.mean(valid_points_y) if valid_points_y else keypoints[:, 0, 1]
    htps_keypoints[:, 0, 1] = keypoints[:, 0, 1] - 4 * abs(mean_y - keypoints[:, 0, 1])

    # Thorax (8)
    thorax_points = keypoints[:, 5:7, :][~np.all(keypoints[:, 5:7, :] == 0, axis=2)]
    htps_keypoints[:, 1, :] = np.mean(thorax_points, axis=0) if thorax_points.size else 0.0
    htps_keypoints[:, 1, :] += (keypoints[:, 0, :] - htps_keypoints[:, 1, :]) / 3

    # Pelvis (0)
    pelvis_points = keypoints[:, 11:13, :]
    htps_keypoints[:, 2, :] = (
        np.mean(pelvis_points, axis=1) if np.any(pelvis_points != 0) else 0.0
    )

    # Spine (7)
    spine_points = keypoints[:, [5, 6, 11, 12], :]
    htps_keypoints[:, 3, :] = (
        np.mean(spine_points, axis=1) if np.any(spine_points != 0) else 0.0
    )

    # 최종 h36m 포맷에 맞게 keypoints 설정
    keypoints_h36m[:, spple_keypoints, :] = htps_keypoints
    keypoints_h36m[:, h36m_coco_order, :] = keypoints[:, coco_order, :]

    # Spine 조정
    keypoints_h36m[:, 8, 1] -= (np.mean(keypoints[:, 1:3, 1], axis=1, dtype=np.float32) - keypoints[:, 0, 1]) * 2 / 3

    # 유효한 프레임 찾기
    valid_frames = np.where(np.sum(keypoints_h36m.reshape(-1, 34), axis=1) != 0)[0]
    
    return keypoints_h36m, valid_frames



def h36m_coco_format(keypoints, scores):
    scores = scores.reshape(1,17)
    h36m_kpts = []
    h36m_scores = []
    valid_frames = []

    new_score = np.zeros_like(scores, dtype=np.float32)

    kpts, valid_frame = coco_h36m(keypoints)
    h36m_kpts.append(kpts)
    valid_frames.append(valid_frame)

    new_score[:, h36m_coco_order] = scores[:, coco_order]
    new_score[:, 0] = np.mean(scores[:, [11, 12]], axis=1, dtype=np.float32)
    new_score[:, 8] = np.mean(scores[:, [5, 6]], axis=1, dtype=np.float32)
    new_score[:, 7] = np.mean(new_score[:, [0, 8]], axis=1, dtype=np.float32)
    new_score[:, 10] = np.mean(scores[:, [1, 2, 3, 4]], axis=1, dtype=np.float32)

    h36m_scores.append(new_score)

    h36m_kpts = np.asarray(h36m_kpts, dtype=np.float32).reshape(-1,17,2)
    h36m_scores = np.asarray(h36m_scores, dtype=np.float32).reshape(-1,17,1)

    return h36m_kpts, h36m_scores, valid_frames