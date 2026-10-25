import cv2
import threading
import time

class CameraHandler:
    """
    Manages multiple camera streams using threading to prevent blocking the main application.
    This is crucial for handling multiple video inputs smoothly.
    """
    def __init__(self, camera_indices):
        """
        Initializes the camera handler.
        Args:
            camera_indices (list): A list of integers representing the camera indices to use (e.g., [0, 1, 2, 3]).
        """
        self.camera_indices = camera_indices
        self.captures = [None] * len(camera_indices)
        self.frames = [None] * len(camera_indices)
        self.is_running = False
        self.threads = []

        for i, index in enumerate(self.camera_indices):
            try:
                cap = cv2.VideoCapture(index)
                if not cap.isOpened():
                    print(f"Warning: Could not open camera at index {index}.")
                    self.captures[i] = None
                else:
                    self.captures[i] = cap
                    print(f"Successfully opened camera at index {index}.")
            except Exception as e:
                print(f"Error opening camera {index}: {e}")
                self.captures[i] = None

    def _read_frames(self, cam_index_in_list):
        """
        The target function for each thread. Continuously reads frames from a camera.
        Args:
            cam_index_in_list (int): The position of the camera in the self.captures list.
        """
        capture = self.captures[cam_index_in_list]
        while self.is_running and capture is not None and capture.isOpened():
            ret, frame = capture.read()
            if ret:
                self.frames[cam_index_in_list] = frame
            else:
                # If reading fails, wait a bit before retrying
                time.sleep(0.1)
        print(f"Stopping thread for camera index {self.camera_indices[cam_index_in_list]}.")

    def start(self):
        """
        Starts the threads for reading from all cameras.
        """
        if self.is_running:
            print("Camera handler is already running.")
            return

        self.is_running = True
        for i in range(len(self.captures)):
            if self.captures[i] is not None:
                thread = threading.Thread(target=self._read_frames, args=(i,), daemon=True)
                self.threads.append(thread)
                thread.start()
        print("Camera reading threads started.")

    def get_frames(self):
        """
        Returns the latest frames from all cameras.
        Returns:
            list: A list of the most recent frames (as numpy arrays).
                  If a camera failed, its spot in the list will be None.
        """
        return self.frames

    def stop(self):
        """
        Stops all threads and releases all camera resources.
        """
        print("Stopping camera handler...")
        self.is_running = False
        for thread in self.threads:
            thread.join(timeout=1) # Wait for threads to finish

        for cap in self.captures:
            if cap is not None:
                cap.release()
        print("All cameras released.")
