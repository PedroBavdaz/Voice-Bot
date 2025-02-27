import pyaudio
import os
import wave
import keyboard
import time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel

# Audio recording constants
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "rec"
OUTPUT_TEXT_FILE = "usertxt.txt"
THRESHOLD = 1.5
SILENCE_DURATION = 0.6
GRACE_CHUNKS = int(RATE / CHUNK * SILENCE_DURATION)

# Initialize audio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialize Faster Whisper model
model = WhisperModel("medium.en")  # Adjust to your model size

print("Press space to record")
keyboard.wait('space')
print("Recording..")
time.sleep(0.2)


def get_rms(data):
    """Calculate RMS volume from raw audio data."""
    samples = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples**2))
    return rms


def transcribe_audio(file_path):
    """Transcribe an audio file using Faster WhisperModel and append it to a single file."""
    try:
        print(f"Transcribing {file_path}...")

        # Check if file exists before transcribing
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return

        # Perform transcription
        segments, info = model.transcribe(file_path)
        transcription = " ".join([segment.text for segment in segments])
        print(f"Transcription for {file_path}:\n{transcription}")

        # Append transcription to the output file
        with open(OUTPUT_TEXT_FILE, "a") as f:
            f.write(transcription + "\n")
        print(f"Appended transcription to {OUTPUT_TEXT_FILE}")

    except Exception as e:
        print(f"Error while processing {file_path}: {e}")
    finally:
        # Delete the file after transcription
        try:
            os.remove(file_path)
            print(f"Deleted {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")


frames = []
file_counter = 0
is_recording = False
silence_counter = 0
total_volume = 0
volume_count = 0

# Use a thread pool for transcription
executor = ThreadPoolExecutor(max_workers=2)

try:
    while True:
        data = stream.read(CHUNK)
        volume = get_rms(data)

        print(f"Volume: {volume:.2f}", end='\r')

        # Constants for minimum recording requirements
        MIN_RECORDING_DURATION = 1  # Minimum duration in seconds
        MIN_AVERAGE_VOLUME = 0.5  # Minimum average volume

        if volume > THRESHOLD:
            if not is_recording:
                is_recording = True
                frames = []
                silence_counter = 0
                total_volume = 0
                volume_count = 0
                print(f"\nStarting new recording {file_counter}")
            frames.append(data)
            total_volume += volume
            volume_count += 1

        elif is_recording:
            silence_counter += 1
            frames.append(data)
            total_volume += volume
            volume_count += 1

            if silence_counter > GRACE_CHUNKS:
                is_recording = False

                # Calculate the duration of the recording (in seconds)
                recording_duration = len(frames) * CHUNK / RATE
                print(f"Recording duration: {recording_duration:.2f} seconds")

                # Calculate the average volume
                avg_volume = total_volume / volume_count if volume_count > 0 else 0
                print(f"Average volume: {avg_volume:.2f}")

                # If the recording is too short or too quiet, discard it
                if recording_duration < MIN_RECORDING_DURATION or avg_volume < MIN_AVERAGE_VOLUME:
                    print(f"Recording too short ({recording_duration:.2f}s) or too quiet (avg volume: {
                          avg_volume:.2f}), discarding...")
                    continue  # Skip saving and transcription for this short/quiet recording

                # Save the recording
                output_filename = f"{OUTPUT_FILENAME}_{file_counter}.wav"
                with wave.open(output_filename, "wb") as waveFile:
                    waveFile.setnchannels(CHANNELS)
                    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                    waveFile.setframerate(RATE)
                    waveFile.writeframes(b''.join(frames))
                print(f"Saved {output_filename}")

                # Submit the transcription task
                executor.submit(transcribe_audio, output_filename)
                print("Submitting the transcription task")
                file_counter += 1

        # Stop recording when space is pressed
        if keyboard.is_pressed('space'):
            print("\nStopping recording")
            time.sleep(0.2)
            break

except KeyboardInterrupt:
    print("\nRecording interrupted")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()
    executor.shutdown(wait=False)

print("Recording session ended.")
