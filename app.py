from flask import Flask, jsonify, request
import sqlite3

app = Flask(__name__)
DB_FILE = 'sensors.db'

def query_db(query, args=(), one=False):
    """DB에 쿼리를 보내고 결과를 가져오는 함수"""
    try:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query, args)
        rv = cur.fetchall()
        conn.close()
        return (rv[0] if rv else None) if one else rv
    except sqlite3.Error as e:
        print(f"Database query error: {e}")
        return None if one else []

@app.route('/')
def index():
    return "<h1>Arduino Sensor Server is running!</h1>"

# 최신 센서 데이터 1개
@app.route('/api/sensors/latest', methods=['GET'])
def get_latest_data():
    reading = query_db("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1", one=True)
    if reading:
        return jsonify(dict(reading))
    return jsonify({"error": "No data available"}), 404

# 센서 데이터 이력
@app.route('/api/sensors/history', methods=['GET'])
def get_history_data():
    timeframe = request.args.get('timeframe', '1h')

    if timeframe == '24h':
        query = "SELECT * FROM readings WHERE timestamp >= datetime('now', '-1 day') ORDER BY timestamp ASC"
    elif timeframe == '7d':
        query = "SELECT * FROM readings WHERE timestamp >= datetime('now', '-7 days') ORDER BY timestamp ASC"
    else: # 기본값 1h
        query = "SELECT * FROM readings WHERE timestamp >= datetime('now', '-1 hour') ORDER BY timestamp ASC"

    readings = query_db(query)
    data = [dict(row) for row in readings]
    return jsonify(data)

if __name__ == '__main__':
    # 외부(Next.js)에서 접속 가능하도록 0.0.0.0으로 설정
    app.run(host='0.0.0.0', port=5000)