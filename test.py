import cv2

# 0번부터 시도해봅니다. 안 되면 1, 2로 바꿔보세요.
CAM_INDEX = 10

print(f"Trying to open camera index: {CAM_INDEX}...")
cap = cv2.VideoCapture(CAM_INDEX)

# 해상도 설정 (C270 지원 해상도)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera!")
    print("Try changing CAM_INDEX to 1 or 2.")
else:
    print("✅ Camera opened successfully!")
    ret, frame = cap.read()
    if ret:
        print("✅ Frame captured! (Camera is working)")
        cv2.imwrite("test_image.jpg", frame)
        print("Saved 'test_image.jpg'. Check this file.")
    else:
        print("❌ Camera opened but failed to read frame.")

cap.release()