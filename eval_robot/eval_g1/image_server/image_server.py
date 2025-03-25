import cv2
import zmq
import time
import struct
from collections import deque

class ImageServer:
    def __init__(self, fps = 30, port = 5555, Unit_Test = False):
        self.fps = fps
        self.port = port
        self.enable_performance_eval = Unit_Test

        # initiate binocular camera: 480 * (640 * 2)
        self.cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640 * 2)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # set ZeroMQ context and socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.bind(f"tcp://*:{self.port}")

        if self.enable_performance_eval:
            self._init_performance_metrics()

        print("Image server has started, waiting for client connections...")
        print(f"resolution: width is {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}, height is {self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}\n")

    def _init_performance_metrics(self):
        self.frame_count = 0  # Total frames sent
        self.time_window = 1.0  # Time window for FPS calculation (in seconds)
        self.frame_times = deque()  # Timestamps of frames sent within the time window
        self.start_time = time.time()  # Start time of the streaming

    def _update_performance_metrics(self, current_time):
        # Add current time to frame times deque
        self.frame_times.append(current_time)
        # Remove timestamps outside the time window
        while self.frame_times and self.frame_times[0] < current_time - self.time_window:
            self.frame_times.popleft()
        # Increment frame count
        self.frame_count += 1

    def _print_performance_metrics(self, current_time):
        if self.frame_count % 30 == 0:
            elapsed_time = current_time - self.start_time
            real_time_fps = len(self.frame_times) / self.time_window
            print(f"[Image Server] Real-time FPS: {real_time_fps:.2f}, Total frames sent: {self.frame_count}, Elapsed time: {elapsed_time:.2f} sec")
        
    def _close(self):
        self.cap.release()
        self.socket.close()
        self.context.term()
        print("[Image Server] The server has been closed.")

    def send_process(self):
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("[Image Server] Frame read is error.")
                    break

                ret, buffer = cv2.imencode('.jpg', frame)
                if not ret:
                    print("[Image Server] Frame imencode is failed.")
                    continue

                jpg_bytes = buffer.tobytes()

                if self.enable_performance_eval:
                    timestamp = time.time()
                    frame_id = self.frame_count
                    header = struct.pack('dI', timestamp, frame_id)  # 8-byte double, 4-byte unsigned int
                    message = header + jpg_bytes
                else:
                    message = jpg_bytes
                
                self.socket.send(message)

                if self.enable_performance_eval:
                    current_time = time.time()
                    self._update_performance_metrics(current_time)
                    self._print_performance_metrics(current_time)

        except KeyboardInterrupt:
            print("[Image Server] Interrupted by user.")
        finally:
            self._close()

if __name__ == "__main__":
    # server = ImageServer(Unit_Test = True)   # test
    server = ImageServer(Unit_Test = False)  # deployment
    server.send_process()