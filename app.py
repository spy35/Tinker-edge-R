from flask import Flask, jsonify, request
import sqlite3

app = Flask(__name__)
DB_FILE = 'sensors.db'

def query_db(query, args=(), one=False):
    """DB에 쿼리를 보내고 결과를 가져오는 함수"""
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute(query, args)
    rv = cur.fetchall()
    conn.close()
    return (rv[0] if rv else None) if one else rv

@app.route('/')
def index():
    return "<h1>Arduino Sensor Server is running!</h1>"

# 최신 센서 데이터 N개를 가져오는 API
@app.route('/api/sensors/latest', methods=['GET'])
def get_latest_data():
    """가장 최근의 센서 데이터 하나만 반환합니다."""
    reading = query_db("SELECT * FROM readings ORDER BY timestamp DESC LIMIT 1", one=True)
    if reading:
        return jsonify(dict(reading))
    return jsonify({"error": "No data available"}), 404

# 특정 기간의 센서 데이터 이력을 가져오는 API
@app.route('/api/sensors/history', methods=['GET'])
def get_history_data():
    """지정된 시간 범위의 센서 데이터 이력을 반환합니다."""
    timeframe = request.args.get('timeframe', '1h') # 기본값: 1시간

    # 기간에 따라 다른 쿼리 실행 (예시)
    if timeframe == '24h':
        # 지난 24시간 데이터
        query = "SELECT * FROM readings WHERE timestamp >= datetime('now', '-1 day') ORDER BY timestamp ASC"
    elif timeframe == '7d':
        # 지난 7일 데이터
        query = "SELECT * FROM readings WHERE timestamp >= datetime('now', '-7 days') ORDER BY timestamp ASC"
    else:
        # 지난 1시간 데이터 (기본)
        query = "SELECT * FROM readings WHERE timestamp >= datetime('now', '-1 hour') ORDER BY timestamp ASC"

    readings = query_db(query)
    data = [dict(row) for row in readings]
    return jsonify(data)

if __name__ == '__main__':
    # 외부에서 접속 가능하도록 host='0.0.0.0'으로 설정
    app.run(host='0.0.0.0', port=5000)