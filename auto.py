import subprocess, time, os
from pathlib import Path

TASK_FILE = "./practice.txt"
AVD_NAME = "Pixel_9"
SNAPSHOT_NAME = r"snap_2025-07-21_15-17-21"
APP_NAME = "com.google.android.apps.maps"
METHOD = "baseline"
TASK_APP = "google_maps"
# TASK_NUMBER = 
PAV_CLIENT_SCRIPT = "./client/client.py"
IMAGE_BASE_PATH = f"./{METHOD}/1/{TASK_APP}"

ADB_CMD = "adb"
EMULATOR_CMD = r"C:\Users\audgj\AppData\Local\Android\Sdk\emulator\emulator.exe"

def start_emulator():
    print(f"Starting emulator {AVD_NAME}...")
    subprocess.Popen([EMULATOR_CMD, "-avd", AVD_NAME, "-snapshot", SNAPSHOT_NAME, "-no-boot-anim", "-no-snapshot-save"])
    print("Waiting for device...")
    subprocess.run([ADB_CMD, "wait-for-device"], check=True)

    while True:
        result = subprocess.run([ADB_CMD, "shell", "getprop", "sys.boot_completed"],
                                stdout=subprocess.PIPE, text=True)
        if result.stdout.strip() == "1":
            break
        time.sleep(1)
    print("Emulator booted.")

def stop_emulator():
    print("Stopping emulator...")
    subprocess.run([ADB_CMD, "emu", "kill"])

def main():
    with open(TASK_FILE, "r", encoding="utf-8") as f:
        tasks = [line.strip() for line in f if line.strip()]

    for idx, task in enumerate(tasks):
        print(f"---- task #{idx}: {task}")
        start_emulator()
        time.sleep(10)

        current_image_path = Path(IMAGE_BASE_PATH) / str(idx)
        os.makedirs(current_image_path, exist_ok=True)

        subprocess.run([
            "python", PAV_CLIENT_SCRIPT,
            "--method", METHOD,
            "--task_number", str(idx),
            "--task", task,
            "--image_path", str(current_image_path),
            "--app_name", TASK_APP
        ], check=True)

        stop_emulator()

    print("All tasks completed.")

if __name__ == "__main__":
    main()
