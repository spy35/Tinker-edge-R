import serial
import sqlite3
import time

SERIAL_PORT = '/dev/ttyACM0'
BAUD_RATE = 9600

DB_FILE = 'sensors.db'

def setup_database():
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS readings (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   timestamp DATETIME DEFAULT CURRENT_TIMESTAMP, 
                   temperature REAL, 
                   humidity REAL)
                   ''')
    conn.commit()
    conn.close()

def main():
    setup_database()

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} BAUD.")
    except serial.SerialException as e:
        print(f"Error connecting to serial port: {e}")
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    while True:
        try:
            line = ser.readline().decode('utf-8').strip()

            if line:
                print(f"Receiver: {line}")
                parts = line.split(',')
                if len(parts) == 2:
                    temp = float(parts[0])
                    humi = float(parts[1])

                    cursor.execute("INSERT INTO readings (temperature, humidity) VALUES (?, ?)", (temp, humi))
                    conn.commit()
                    print(f"Saver to DB: Temp = {temp}, Humi = {humi}")

        except Exception as e:
            print(f"an error occurred: {e}")
            time.sleep(2)

if  __name__ == '__main__':
    main()