import pyaudio
import numpy as np
import time
from tqdm import tqdm
import os

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1  # Mono audio
RATE = 44100
CHUNK = 1024

# Function to record and display live volume in the command line


def record_and_display_volume():
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording... Press Ctrl+C to stop.")

    # Set up the progress bar
    with tqdm(total=50, desc="Volume", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ({rate_fmt})", ncols=70) as pbar:
        try:
            while True:
                os.system('cls')
                data = stream.read(CHUNK)
                audio_data = np.frombuffer(data, dtype=np.int16)

                # Calculate the volume (RMS)
                volume = np.sqrt(np.mean(np.square(audio_data)))
                # Debugging line to check the volume value
                print(f"Raw volume value: {volume}")

                # Check if volume is valid (not NaN or zero)
                if np.isnan(volume) or volume == 0:
                    continue  # Skip this loop iteration if volume is invalid

                # Map the volume to a progress bar size (e.g., max size 50)
                # Decrease divisor to make the volume bar more sensitive
                volume_bar = int(volume / 2)
                volume_bar = min(volume_bar, 50)  # Limit max bar size to 50

                # Debugging line to ensure the bar size is being updated
                # Debugging the scaled volume for progress bar
                print(f"Volume Bar (scaled): {volume_bar}")

                # Update the progress bar with the current volume
                pbar.n = volume_bar  # Set the current position of the bar
                pbar.last_print_n = volume_bar
                pbar.update(0)  # Update without changing the bar size

                # Sleep to reduce the update frequency (optional)
                time.sleep(0.1)

        except KeyboardInterrupt:
            print("\nRecording stopped.")

    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    audio.terminate()


# Start the program
record_and_display_volume()
