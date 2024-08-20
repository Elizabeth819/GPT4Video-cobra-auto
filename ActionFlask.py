from flask import Flask, request, jsonify  
from flask_socketio import SocketIO, send, emit  
import eventlet  
import ActionSummary as acsum
eventlet.monkey_patch()  
  
app = Flask(__name__)  
app.config['SECRET_KEY'] = 'secret!'  
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')  
  
# Wrap the AnalyzeVideo function to use WebSockets  
def AnalyzeVideoSocket(video_path, frame_interval, frames_per_interval,socketio, face_rec=False):  
    # Call the original AnalyzeVideo function  
    final_arr=acsum.AnalyzeVideo(video_path, frame_interval, frames_per_interval,socketio, face_rec)  
  
    # Send the final_arr to the client using WebSocket  
    socketio.emit('actionsummary', {'data': final_arr}, namespace='/test')  
  
@app.route('/analyze', methods=['POST'])  
def analyze_video():  
    # Get the video_path, frame_interval and frames_per_interval from request data  
    video_path = request.json.get('video_path')  
    frame_interval = request.json.get('frame_interval')  
    frames_per_interval = request.json.get('frames_per_interval')  
    face_rec = request.json.get('face_rec', False)  
  
    # Start a new thread to analyze the video and send updates to the client  
    socketio.start_background_task(AnalyzeVideoSocket, video_path, frame_interval, frames_per_interval, face_rec)  
  
    return jsonify({'status': 'started'})  
  
@socketio.on('connect', namespace='/test')  
def test_connect():  
    print('Client connected')  
  
@socketio.on('disconnect', namespace='/test')  
def test_disconnect():  
    print('Client disconnected')  
  
if __name__ == '__main__':  
    socketio.run(app, debug=True)  
