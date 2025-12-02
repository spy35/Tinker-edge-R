import cv2
import time
import threading
import numpy as np
import sqlite3
from flask import Flask, Response, jsonify

# --- [중요] rknnlite 라이브러리 임포트 ---
# 만약 여기서 에러가 난다면 rknnlite가 설치되지 않은 것입니다.
try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("Error: 'rknnlite' library not found!")
    print("Please install rknn_toolkit_lite whl package first.")
    exit(1)

# --- 설정 ---
RKNN_MODEL_PATH = './yolov5s.rknn'  # rknn 파일이 같은 폴더에 있어야 합니다
IMG_SIZE = (640, 640)
DB_FILE = 'sensors.db'
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# --- YOLOv5 클래스 및 앵커 정의 (postprocess.py 내용 통합) ---
CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
           "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
           "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
           "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
           "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
           "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
           "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
           "hair drier", "toothbrush")

ANCHORS = [[[10, 13], [16, 30], [33, 23]], 
           [[30, 61], [62, 45], [59, 119]], 
           [[116, 90], [156, 198], [373, 326]]]

# --- 전역 변수 ---
output_frame = None
lock = threading.Lock()
current_person_count = 0
npu_initialized = False

app = Flask(__name__)

# --- 헬퍼 함수: 좌표 변환 ---
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# [수정된 post_process 함수]
def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []

    for pred in input_data:
        # 1. 데이터 크기로 그리드 사이즈 및 스트라이드 역산
        flat_size = pred.size
        # YOLOv5 출력: (3개 앵커) * (85개 정보) * (H) * (W)
        # H*W = size / (3*85)
        grid_len = flat_size // (3 * 85)
        grid_size = int(np.sqrt(grid_len))
        
        stride = int(IMG_SIZE[0] / grid_size)

        # 스트라이드에 맞는 앵커 세트 선택 (P3, P4, P5)
        if stride == 8: anchor_idx = 0
        elif stride == 16: anchor_idx = 1
        elif stride == 32: anchor_idx = 2
        else: continue # 알 수 없는 크기는 무시
            
        # 현재 스케일의 앵커들 (Shape: [3, 2])
        current_anchors = np.array(anchors[anchor_idx])

        # 2. 데이터 모양 변경 (3, Grid, Grid, 85)
        pred = pred.reshape((3, grid_size, grid_size, 85))

        # 3. 신뢰도(Confidence) 필터링
        # Box confidence * Class probability가 임계값을 넘는 것만 선택
        # 속도 최적화를 위해 먼저 box_conf만 검사할 수도 있으나, 여기선 정석대로 함
        box_conf = 1 / (1 + np.exp(-pred[..., 4]))
        pos = np.where(box_conf > OBJ_THRESH)
        
        # 감지된 것이 없으면 다음 레이어로
        if pos[0].shape[0] == 0:
            continue

        # 필터링된 데이터만 가져오기 (Shape: [N, 85])
        pred = pred[pos]

        # 4. 좌표 복원 (Decode)
        # pos[0]: anchor indices (0~2), pos[1]: grid_y, pos[2]: grid_x
        
        # 그리드 좌표 (N, 2)
        grid_xy = np.stack((pos[2], pos[1]), axis=1)
        
        # 예측된 오프셋 (Sigmoid 적용)
        pred_xy = 1 / (1 + np.exp(-pred[..., 0:2])) * 2. - 0.5
        
        # 최종 x, y 좌표 계산: (offest + grid) * stride
        x = (pred_xy + grid_xy) * stride
        
        # w, h 계산: (sigmoid(val) * 2)^2 * anchors
        # 해당 감지 건에 맞는 앵커 가져오기 (pos[0]을 인덱스로 사용)
        anchors_selected = current_anchors[pos[0]]
        
        pred_wh = (1 / (1 + np.exp(-pred[..., 2:4])) * 2) ** 2
        w = pred_wh * anchors_selected

        # 5. 클래스 점수 계산
        class_conf = 1 / (1 + np.exp(-pred[..., 5:]))
        class_id = np.argmax(class_conf, axis=-1)
        class_prob = np.max(class_conf, axis=-1)

        # 6. 결과 합치기
        # xywh (중심x, 중심y, 너비, 높이) -> xyxy (좌상단, 우하단) 변환
        xywh = np.concatenate((x, w), axis=-1)
        xyxy = xywh2xyxy(xywh)
        
        boxes.append(xyxy)
        scores.append(box_conf[pos] * class_prob) # 최종 점수 = 박스확률 * 클래스확률
        classes_conf.append(class_id)

    if not boxes: return None, None, None

    # 모든 레이어의 결과 병합
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    classes_conf = np.concatenate(classes_conf)

    # 7. NMS (겹치는 박스 제거)
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), OBJ_THRESH, NMS_THRESH)

    if len(indices) > 0:
        return boxes[indices.flatten()], classes_conf[indices.flatten()], scores[indices.flatten()]
    return None, None, None

# --- DB 조회 함수 ---
def query_db(query, args=(), one=False):
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, args)
        rv = cur.fetchall()
        conn.close()
        return (rv[0] if rv else None) if one else rv
    except sqlite3.Error:
        return None

# --- NPU 엔진 클래스 ---
class YoloApp:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        print("--> Loading RKNN model")
        if self.rknn.load_rknn(model_path) != 0:
            print("Load RKNN model failed")
            exit(1)
        print("--> Init runtime (RK3399Pro)")
        if self.rknn.init_runtime(target='rk3399pro') != 0:
            print("Init runtime environment failed")
            exit(1)

    def infer(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        input_data = np.expand_dims(img, axis=0)
        outputs = self.rknn.inference(inputs=[input_data])
        return outputs

    def release(self):
        self.rknn.release()

# --- 카메라 스레드 ---
def camera_thread_func():
    global output_frame, current_person_count, npu_initialized
    
    try:
        yolo_engine = YoloApp(RKNN_MODEL_PATH)
        npu_initialized = True
    except Exception as e:
        print(f"NPU Init Failed: {e}")
        return

    cap = cv2.VideoCapture(10)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Camera open failed!")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # 1. 추론
        outputs = yolo_engine.infer(frame)
        
        # 2. 후처리
        boxes, classes, scores = post_process(outputs, ANCHORS)
        
        p_count = 0
        if boxes is not None:
            for box, cls_id, score in zip(boxes, classes, scores):
                # 사람(Class ID 0)만 카운트
                if int(cls_id) == 0:
                    p_count += 1
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # 좌표 변환 (640x640 -> 원본 해상도)
                    h, w, _ = frame.shape
                    x1 = int(x1 * w / IMG_SIZE[0])
                    y1 = int(y1 * h / IMG_SIZE[1])
                    x2 = int(x2 * w / IMG_SIZE[0])
                    y2 = int(y2 * h / IMG_SIZE[1])

                    # 그리기
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {score:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        current_person_count = p_count

        # 3. 프레임 업데이트
        with lock:
            output_frame = frame.copy()
        
        time.sleep(0.01)

# --- Flask 라우트 ---
@app.route('/')
def index():
    return "<h1>Tinker Edge R - AI Sensor Gateway</h1>"

@app.route('/api/sensors/latest', methods=['GET'])
def get_latest_data():
    response = {}
    
    env = query_db("SELECT temperature, humidity, motion FROM readings WHERE node_type='ENV' ORDER BY timestamp DESC LIMIT 1", one=True)
    if env: response.update(dict(env))
    
    pwr = query_db("SELECT current, power FROM readings WHERE node_type='PWR' ORDER BY timestamp DESC LIMIT 1", one=True)
    if pwr: response.update(dict(pwr))
    
    response['people_count'] = current_person_count
    
    return jsonify(response)

@app.route('/api/sensors/history', methods=['GET'])
def get_history_data():
    query = "SELECT * FROM readings ORDER BY timestamp DESC LIMIT 100"
    readings = query_db(query)
    return jsonify([dict(row) for row in readings])

def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag:
                continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # 카메라 스레드 시작
    t = threading.Thread(target=camera_thread_func)
    t.daemon = True
    t.start()
    
    # 웹 서버 시작
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)