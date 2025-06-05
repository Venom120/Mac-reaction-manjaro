import time
import subprocess
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.check_modprobe()
        self.start_script()

    def check_modprobe(self):
        try:
            # Check if module is already loaded
            lsmod = subprocess.check_output(['lsmod']).decode()
            if 'v4l2loopback' not in lsmod:
                print("Loading v4l2loopback kernel module...")
                subprocess.run([
                    'sudo', 'modprobe', 'v4l2loopback',
                    'devices=1',
                    'video_nr=10',
                    'card_label=Reactions',
                    'exclusive_caps=1'
                ], check=True)
            else:
                print("v4l2loopback already loaded.")
        except subprocess.CalledProcessError as e:
            print("Failed to load v4l2loopback module. Check permissions or kernel setup.")
            sys.exit(1)

    def start_script(self):
        print("Starting main.py...")
        python_exec = sys.executable  # use same python running this
        self.process = subprocess.Popen([python_exec, 'main.py'])

    def stop_script(self):
        if self.process and self.process.poll() is None:
            print("Stopping main.py...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.py',)):
            print(f"Detected change in {event.src_path}. Restarting...")
            self.stop_script()
            self.start_script()

if __name__ == "__main__":
    watch_path = "."
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
import time
import subprocess
import sys
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ChangeHandler(FileSystemEventHandler):
    def __init__(self):
        self.process = None
        self.check_modprobe()
        self.start_script()

    def check_modprobe(self):
        try:
            # Check if module is already loaded
            lsmod = subprocess.check_output(['lsmod']).decode()
            if 'v4l2loopback' not in lsmod:
                print("Loading v4l2loopback kernel module...")
                subprocess.run([
                    'modprobe', 'v4l2loopback',
                    'devices=1',
                    'video_nr=10',
                    'card_label=Reactions',
                    'exclusive_caps=1'
                ], check=True)
            else:
                print("v4l2loopback already loaded.")
        except subprocess.CalledProcessError as e:
            print("Failed to load v4l2loopback module. Check permissions or kernel setup.")
            sys.exit(1)

    def start_script(self):
        print("Starting main.py...")
        python_exec = sys.executable  # use same python running this
        self.process = subprocess.Popen([python_exec, 'main.py'])

    def stop_script(self):
        if self.process and self.process.poll() is None:
            print("Stopping main.py...")
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith(('.py',)):
            print(f"Detected change in {event.src_path}. Restarting...")
            self.stop_script()
            self.start_script()

if __name__ == "__main__":
    watch_path = "."
    event_handler = ChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, watch_path, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
