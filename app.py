import cv2
import time
import threading
import numpy as np
import sqlite3
from flask import Flask, Response, jsonify

# --- [중요] rknnlite 라이브러리 임포트 ---
try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("Error: 'rknnlite' library not found!")
    # 테스트를 위해 로컬 PC에서 돌릴 땐 아래 줄 주석 처리 또는 모의 객체 사용 필요
    # exit(1) 
    pass

# --- 설정 ---
RKNN_MODEL_PATH = './yolov5s.rknn'
IMG_SIZE = (640, 640)
DB_FILE = 'sensors.db'
OBJ_THRESH = 0.25
NMS_THRESH = 0.45

# --- [추가] 규칙 기반 제어 설정값 ---
DELAY_TIME = 300  # 5분 (300초)
TEMP_THRESHOLD = 26.0
POWER_THRESHOLD = 500.0

# --- 전역 변수 ---
output_frame = None
lock = threading.Lock()
current_person_count = 0
npu_initialized = False

# [상태 관리 변수]
system_state = {
    "mode": "ACTIVE",      # ACTIVE, HOLD, ECO
    "message": "시스템 시작 중...",
    "alert_level": "normal", # normal, warning, critical
    "last_motion_time": time.time()
}

app = Flask(__name__)

# --- (기존 코드 유지) YOLO 관련 함수들: xywh2xyxy, post_process, YoloApp 등 ---
# ... (여기에 기존 app.py의 post_process, YoloApp 클래스 코드가 그대로 들어갑니다) ...
# (코드가 길어 생략하지만, 사용자가 올린 파일의 내용과 동일하게 유지하세요)

# --- 헬퍼 함수: 좌표 변환 ---
def xywh2xyxy(x):
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

# [기존 post_process 함수 유지]
def post_process(input_data, anchors):
    boxes, scores, classes_conf = [], [], []
    for pred in input_data:
        flat_size = pred.size
        grid_len = flat_size // (3 * 85)
        grid_size = int(np.sqrt(grid_len))
        stride = int(IMG_SIZE[0] / grid_size)
        if stride == 8: anchor_idx = 0
        elif stride == 16: anchor_idx = 1
        elif stride == 32: anchor_idx = 2
        else: continue
            
        current_anchors = np.array(anchors[anchor_idx])
        pred = pred.reshape((3, grid_size, grid_size, 85))
        box_conf = 1 / (1 + np.exp(-pred[..., 4]))
        pos = np.where(box_conf > OBJ_THRESH)
        if pos[0].shape[0] == 0: continue
        pred = pred[pos]
        grid_xy = np.stack((pos[2], pos[1]), axis=1)
        pred_xy = 1 / (1 + np.exp(-pred[..., 0:2])) * 2. - 0.5
        x = (pred_xy + grid_xy) * stride
        anchors_selected = current_anchors[pos[0]]
        pred_wh = (1 / (1 + np.exp(-pred[..., 2:4])) * 2) ** 2
        w = pred_wh * anchors_selected
        class_conf = 1 / (1 + np.exp(-pred[..., 5:]))
        class_id = np.argmax(class_conf, axis=-1)
        class_prob = np.max(class_conf, axis=-1)
        xywh = np.concatenate((x, w), axis=-1)
        xyxy = xywh2xyxy(xywh)
        boxes.append(xyxy)
        scores.append(box_conf[pos] * class_prob)
        classes_conf.append(class_id)

    if not boxes: return None, None, None
    boxes = np.concatenate(boxes)
    scores = np.concatenate(scores)
    classes_conf = np.concatenate(classes_conf)
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

# --- [수정 완료] 규칙 기반 제어 로직 스레드 ---
def control_logic_thread():
    global system_state, current_person_count
    
    # 에코 모드 내부 상태 관리용 변수
    eco_sub_state = "IDLE" # IDLE -> VERIFYING -> COOLDOWN
    verify_start_time = 0
    
    print("--- [Logic Thread] Control System Started ---")

    while True:
        try:
            # 1. 센서 데이터 읽기
            env = query_db("SELECT temperature, motion FROM readings WHERE node_type='ENV' ORDER BY timestamp DESC LIMIT 1", one=True)
            pwr = query_db("SELECT power FROM readings WHERE node_type='PWR' ORDER BY timestamp DESC LIMIT 1", one=True)
            
            temp = env['temperature'] if env else 24.0
            pir_detected = (env['motion'] == 1) if env else False
            power = pwr['power'] if pwr else 0.0
            
            yolo_person = current_person_count
            is_occupied = (yolo_person > 0)
            
            current_time = time.time()
            time_diff = current_time - system_state["last_motion_time"]
            
            # --- 상태 머신 (State Machine) ---

            # [MODE: ACTIVE]
            if system_state["mode"] == "ACTIVE":
                eco_sub_state = "IDLE" 
                
                if is_occupied or pir_detected:
                    system_state["last_motion_time"] = current_time
                    if temp > TEMP_THRESHOLD:
                        # 소수점 제거 또는 고정하여 불필요한 변경 방지
                        system_state["message"] = f"[제어 권장] 냉방기 필요: 현재 {int(temp)}°C (기준 {int(TEMP_THRESHOLD)}°C 초과)"
                        system_state["alert_level"] = "warning"
                    else:
                        system_state["message"] = f"[초록색] 재실 확인: {yolo_person}명 감지됨. (Mode: Active)"
                        system_state["alert_level"] = "normal"
                else:
                    system_state["mode"] = "HOLD"
                    system_state["message"] = "미재실 감지. 절전 모드 전환까지 5분간 홀드됩니다."
                    system_state["alert_level"] = "normal"

            # [MODE: HOLD]
            elif system_state["mode"] == "HOLD":
                if is_occupied or pir_detected:
                    system_state["mode"] = "ACTIVE"
                    system_state["last_motion_time"] = current_time
                    system_state["message"] = "[초록색] 재실 복구: 타이머 초기화 (Active 유지)"
                    system_state["alert_level"] = "normal"
                else:
                    # 5분(300초) 경과 -> ECO 모드 전환
                    if time_diff >= DELAY_TIME:
                        system_state["mode"] = "ECO"
                        system_state["message"] = "장시간 미재실! 절전 모드로 전환하세요."
                        system_state["alert_level"] = "critical"
                    
                    # 3분(180초) 경과 -> 2분 남았음을 '한 번만' 알림 (메시지 내용 고정)
                    elif time_diff >= 180:
                        # 매초 숫자가 바뀌지 않도록 고정 텍스트 사용
                        system_state["message"] = "[정보 업데이트] 2분 후 절전모드 전환 예정. (현재 상태 유지 중)"
                        system_state["alert_level"] = "warning"
                    
                    # 3분 미만
                    else:
                        system_state["message"] = "미재실 감지. 절전 모드 전환까지 5분간 홀드됩니다."
                        system_state["alert_level"] = "normal"

            # [MODE: ECO] (이전과 동일하게 유지 - 검증 로직)
            elif system_state["mode"] == "ECO":
                
                # 1단계: 검증 중 (VERIFYING)
                if eco_sub_state == "VERIFYING":
                    elapsed = current_time - verify_start_time
                    
                    if elapsed < 3.0:
                        system_state["message"] = "[정보] 움직임 감지! 재실 여부를 확인합니다. (YOLO 확인 중...)"
                        system_state["alert_level"] = "warning"
                    else:
                        if is_occupied:
                            system_state["mode"] = "ACTIVE"
                            system_state["last_motion_time"] = current_time
                            system_state["message"] = "[초록색] 연구원 복귀 확인! 쾌적 모드로 자동 전환됩니다."
                            system_state["alert_level"] = "normal"
                            eco_sub_state = "IDLE"
                        else:
                            system_state["message"] = "[정보] 비재실 확인. 움직임은 오작동입니다. 절전 모드가 유지됩니다."
                            system_state["alert_level"] = "warning"
                            eco_sub_state = "COOLDOWN"

                # 2단계: 쿨다운 (COOLDOWN)
                elif eco_sub_state == "COOLDOWN":
                    if not pir_detected:
                        eco_sub_state = "IDLE"
                        system_state["message"] = "장시간 미재실! 절전 모드 유지 중."
                        system_state["alert_level"] = "critical"

                # 3단계: 대기 중 (IDLE)
                else:
                    if pir_detected:
                        eco_sub_state = "VERIFYING"
                        verify_start_time = current_time
                        system_state["message"] = "[정보] 움직임 감지! 재실 여부를 확인합니다. (YOLO 확인 중...)"
                        system_state["alert_level"] = "warning"
                    
                    elif power > POWER_THRESHOLD:
                        system_state["message"] = f"[권장] 소비전력 {int(power)}W. 기기를 꺼주세요."
                        system_state["alert_level"] = "critical"
                    else:
                        # 평상시 에코 모드 메시지 유지 (도배 방지)
                        if system_state["message"] != "장시간 미재실! 절전 모드 유지 중.":
                             system_state["message"] = "장시간 미재실! 절전 모드 유지 중."
                             system_state["alert_level"] = "critical"

        except Exception as e:
            print(f"Logic Error: {e}")
        
        time.sleep(1)

# --- 카메라 스레드 (기존 유지 + ANCHORS 정의 필요) ---
CLASSES = ("person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

ANCHORS = [[[10, 13], [16, 30], [33, 23]], [[30, 61], [62, 45], [59, 119]], [[116, 90], [156, 198], [373, 326]]]

class YoloApp:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        self.rknn.load_rknn(model_path)
        self.rknn.init_runtime(target='rk3399pro')
    def infer(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        input_data = np.expand_dims(img, axis=0)
        return self.rknn.inference(inputs=[input_data])
    def release(self):
        self.rknn.release()

def camera_thread_func():
    global output_frame, current_person_count, npu_initialized
    try:
        yolo_engine = YoloApp(RKNN_MODEL_PATH)
        npu_initialized = True
    except Exception as e:
        print(f"NPU Init Failed: {e}")
        return

    cap = cv2.VideoCapture(11) # 카메라 인덱스 확인 필요
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret: continue

        outputs = yolo_engine.infer(frame)
        boxes, classes, scores = post_process(outputs, ANCHORS)
        
        p_count = 0
        if boxes is not None:
            for box, cls_id, score in zip(boxes, classes, scores):
                if int(cls_id) == 0: # person class
                    p_count += 1
                    x1, y1, x2, y2 = box.astype(int)
                    h, w, _ = frame.shape
                    x1 = int(x1 * w / IMG_SIZE[0])
                    y1 = int(y1 * h / IMG_SIZE[1])
                    x2 = int(x2 * w / IMG_SIZE[0])
                    y2 = int(y2 * h / IMG_SIZE[1])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"Person {score:.2f}", (x1, y1-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        current_person_count = p_count
        with lock:
            output_frame = frame.copy()
        time.sleep(0.01)

# --- Flask 라우트 수정 ---
@app.route('/api/sensors/latest', methods=['GET'])
def get_latest_data():
    response = {}
    
    # DB에서 최신 센서값 조회
    env = query_db("SELECT temperature, humidity, motion FROM readings WHERE node_type='ENV' ORDER BY timestamp DESC LIMIT 1", one=True)
    if env: response.update(dict(env))
    
    pwr = query_db("SELECT current, power FROM readings WHERE node_type='PWR' ORDER BY timestamp DESC LIMIT 1", one=True)
    if pwr: response.update(dict(pwr))
    
    # 현재 계산된 시스템 상태 병합
    response['people_count'] = current_person_count
    response['system_mode'] = system_state['mode']
    response['system_message'] = system_state['message']
    response['alert_level'] = system_state['alert_level']
    
    return jsonify(response)

# ... (나머지 video_feed, history 등은 기존 유지) ...
def generate():
    global output_frame
    while True:
        with lock:
            if output_frame is None: continue
            (flag, encodedImage) = cv2.imencode(".jpg", output_frame)
            if not flag: continue
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')
        time.sleep(0.05)

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # 1. 카메라 스레드
    t_cam = threading.Thread(target=camera_thread_func)
    t_cam.daemon = True
    t_cam.start()
    
    # 2. [추가] 제어 로직 스레드
    t_logic = threading.Thread(target=control_logic_thread)
    t_logic.daemon = True
    t_logic.start()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)