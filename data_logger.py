import serial
import sqlite3
import time

# NER-10756 연결 포트 (ls /dev/ttyUSB* 확인)
SERIAL_PORT = '/dev/ttyUSB0' 
BAUD_RATE = 9600
DB_FILE = 'sensors.db'

def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS readings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            node_type TEXT, 
            temperature REAL,
            humidity REAL,
            motion INTEGER,
            current REAL,
            power REAL
        )
    ''')
    conn.commit()
    conn.close()
    print("Database setup complete.")

def main():
    setup_database()
    
    while True:
        try:
            print(f"Connecting to {SERIAL_PORT}...")
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Connected to {SERIAL_PORT}")
            
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            while True:
                try:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                except UnicodeDecodeError:
                    continue

                if line:
                    print(f"Raw: {line}")

                    # [수정됨] 데이터 전처리: 불필요한 디버그 메시지 제거
                    # 아두이노에서 보낸 "Sending via Zigbee:" 같은 문구를 삭제합니다.
                    clean_line = line.replace("Sending via Zigbee:", "").replace("Data Sent:", "").strip()
                    
                    parts = clean_line.split(',')

                    # 1. 전력 데이터 처리 (S2,전류,전력)
                    if len(parts) == 3 and parts[0] == 'S2':
                        try:
                            curr = float(parts[1])
                            pwr = float(parts[2])
                            cursor.execute('''
                                INSERT INTO readings (node_type, current, power) 
                                VALUES (?, ?, ?)
                            ''', ('PWR', curr, pwr))
                            conn.commit()
                            print(f"Saved [PWR]: {pwr}W")
                        except ValueError:
                            print(f"PWR parse error: {clean_line}")

                    # 2. 온습도 데이터 처리 (온도,습도,움직임) -> 숫자로 시작
                    elif len(parts) == 3 and parts[0] != 'S2':
                        try:
                            # 첫번째 데이터가 숫자인지 확인
                            temp = float(parts[0])
                            humi = float(parts[1])
                            motion = int(parts[2])
                            
                            cursor.execute('''
                                INSERT INTO readings (node_type, temperature, humidity, motion) 
                                VALUES (?, ?, ?, ?)
                            ''', ('ENV', temp, humi, motion))
                            conn.commit()
                            print(f"Saved [ENV]: {temp}C, {humi}%")
                        except ValueError:
                            # "S2"도 아니고 숫자도 아니면 노이즈로 간주
                            print(f"ENV parse error (or noise): {clean_line}")
                            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(3)

if __name__ == '__main__':
    main()