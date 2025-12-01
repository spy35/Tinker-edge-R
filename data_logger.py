import serial
import sqlite3
import time

# 게이트웨이 아두이노가 연결된 시리얼 포트
# (Tinker Edge R 터미널에서 `ls /dev/tty*`로 확인. ACM0 또는 USB0 등일 수 있음)
SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600
DB_FILE = 'sensors.db' # 스크립트와 같은 위치에 생성됨

def setup_database():
    """데이터베이스와 테이블을 생성하는 함수 (motion 컬럼 포함)"""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
   
    try:
        # DB 테이블이 없으면 생성
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                temperature REAL,
                humidity REAL,
                motion INTEGER  
            )
        ''')
       
        # 테이블이 이미 존재할 경우 motion 컬럼만 추가 (에러 무시)
        try:
            cursor.execute("ALTER TABLE readings ADD COLUMN motion INTEGER")
        except sqlite3.OperationalError as e:
            if "duplicate column name" in str(e):
                pass # 이미 컬럼이 있으므로 정상
            else:
                print(f"DB alter error: {e}")
               
    except sqlite3.OperationalError as e:
        print(f"DB create error: {e}")
           
    conn.commit()
    conn.close()
    print("Database setup complete. 'motion' column ensured.")

def main():
    setup_database()
   
    while True: # 연결이 끊어져도 계속 재시도
        try:
            print(f"Attempting to connect to {SERIAL_PORT}...")
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
            print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} BAUD.")
           
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            while True:
                line = ser.readline().decode('utf-8').strip()
               
                if line:
                    print(f"Received from Gateway: {line}")
                   
                    # 데이터 파싱 ("온도,습도,움직임")
                    parts = line.split(',')
                    if len(parts) == 3:
                        try:
                            temp = float(parts[0])
                            humi = float(parts[1])
                            motion = int(parts[2])
                           
                            # DB에 3개 데이터 삽입
                            cursor.execute("INSERT INTO readings (temperature, humidity, motion) VALUES (?, ?, ?)",
                                           (temp, humi, motion))
                            conn.commit()
                            print(f"Saved to DB: Temp={temp}, Humi={humi}, Motion={motion}")
                        except ValueError as ve:
                            print(f"Data parsing error: {ve}. Data: '{line}'")
                        except sqlite3.Error as se:
                            print(f"Database error: {se}")
                           
                    elif "Failed to read" in line:
                        print("Received sensor read failure from node.")
                    else:
                        print(f"Received malformed data: '{line}'")

        except serial.SerialException as e:
            print(f"Serial port error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            time.sleep(5)

if __name__ == '__main__':
    main()