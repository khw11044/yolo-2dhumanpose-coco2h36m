from PIL import Image
import os

# 프레임 폴더 경로와 GIF 저장 경로
frames_folder = "./frames"
output_gif = "output.gif"

# frames 폴더 내 모든 jpg 파일 불러오기
image_files = [os.path.join(frames_folder, f) for f in sorted(os.listdir(frames_folder)) if f.endswith(".jpg")]

# 이미지 파일 확인
if not image_files:
    print("Error: No JPG files found in the frames folder.")
    exit()

# 이미지 열기 및 GIF 생성
frames = []
for idx, image_file in enumerate(image_files):
    if idx%3==0:
        img = Image.open(image_file)
        # 이미지 사이즈를 절반으로 조정
        new_size = (img.width // 2, img.height // 2)
        img_resized = img.resize(new_size, Image.Resampling.LANCZOS)
        frames.append(img_resized)

# 첫 프레임을 기준으로 GIF 생성
frames[0].save(
    output_gif,
    format="GIF",
    append_images=frames[1:],  # 첫 프레임 제외 나머지 추가
    save_all=True,            # 모든 프레임 저장
    duration=10,             # 각 프레임 지속 시간 (밀리초)
    loop=0                    # 0은 무한 루프
)

print(f"GIF 파일이 {output_gif}로 저장되었습니다.")