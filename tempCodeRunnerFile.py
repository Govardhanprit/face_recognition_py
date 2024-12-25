def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')