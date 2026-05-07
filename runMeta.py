import csv
import os
import time
from datetime import datetime

import serial
from chirpCor import collect_data

PORT = "COM4"
BAUD_RATE = 115200
CSV_FILE = "experiment_log.csv"
DATA_FOLDER = "data"

os.makedirs(DATA_FOLDER, exist_ok=True)

print("Opening serial connection...")
arduino = serial.Serial(PORT, BAUD_RATE, timeout=2)
time.sleep(2)
print("Serial connected.")


def send_command(cmd: str):
    print(f"Sending command: {cmd}")
    arduino.write((cmd + "\n").encode())

    while True:
        line = arduino.readline().decode(errors="ignore").strip()
        if line:
            print("Arduino said:", line)
        if line == "done":
            print("Move finished.")
            return


def append_csv(row):
    file_exists = os.path.exists(CSV_FILE)

    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "timestamp",
                "direction",
                "move_index",
                "peak_corr",
                "peak_lag",
                "mic_rms",
                "mic_peak",
                "num_samples",
                "save_prefix",
            ])
        writer.writerow(row)


try:
    for i in range(5):
        send_command("FWD")
        print("Collecting forward audio...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prefix = os.path.join(DATA_FOLDER, f"forward_{i+1}_{timestamp}")

        result = collect_data(
            duration=5,
            sample_rate=96000,
            show_plots=False,
            save_prefix=save_prefix,
        )

        print("Forward data collected.")
        append_csv([
            datetime.now().isoformat(),
            "forward",
            i + 1,
            result["peak_corr"],
            result["peak_lag"],
            result["mic_rms"],
            result["mic_peak"],
            result["num_samples"],
            save_prefix,
        ])

    for i in range(5):
        send_command("BWD")
        print("Collecting backward audio...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_prefix = os.path.join(DATA_FOLDER, f"backward_{i+1}_{timestamp}")

        result = collect_data(
            duration=5,
            sample_rate=96000,
            show_plots=False,
            save_prefix=save_prefix,
        )

        print("Backward data collected.")
        append_csv([
            datetime.now().isoformat(),
            "backward",
            i + 1,
            result["peak_corr"],
            result["peak_lag"],
            result["mic_rms"],
            result["mic_peak"],
            result["num_samples"],
            save_prefix,
        ])

finally:
    arduino.close()
    print("Finished and serial closed.")
