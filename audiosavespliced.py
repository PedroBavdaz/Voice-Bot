import pyaudio
import wave
import keyboard
import time
import numpy as np

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
OUTPUT_FILENAME = "rec"
THRESHOLD = 0.6  # Adjust for your environment (RMS value)
SILENCE_DURATION = 0.2  # Time in seconds to wait after volume drops
GRACE_CHUNKS = int(RATE / CHUNK * SILENCE_DURATION)  # Convert to chunks

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

print("Press space to record")
keyboard.wait('space')
print("Recording..")
time.sleep(0.2)


def get_rms(data):
    """Calculate RMS volume from raw audio data."""
    samples = np.frombuffer(data, dtype=np.int16)
    rms = np.sqrt(np.mean(samples**2))
    if rms is not None:
        return rms


frames = []
file_counter = 0
is_recording = False
silence_counter = 0  # Counts silent chunks

try:
    while True:
        data = stream.read(CHUNK)
        volume = get_rms(data)

        # Display volume in real-time
        print(f"Volume: {volume:.2f}", end='\r')

        if volume > THRESHOLD:
            if not is_recording:
                is_recording = True
                frames = []  # Start a new snippet
                silence_counter = 0  # Reset silence counter
                print(f"\nStarting new recording {file_counter}")
            frames.append(data)

        elif is_recording:
            silence_counter += 1
            frames.append(data)
            # If silence duration is reached, save the snippet
            if silence_counter > GRACE_CHUNKS:
                print(str(silence_counter)+"/" + str(GRACE_CHUNKS))
                is_recording = False
                output_filename = f"{OUTPUT_FILENAME}_{file_counter}.wav"
                with wave.open(output_filename, "wb") as waveFile:
                    waveFile.setnchannels(CHANNELS)
                    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                    waveFile.setframerate(RATE)
                    waveFile.writeframes(b''.join(frames))
                print(f"Saved {output_filename}")
                file_counter += 1

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

print("Recording session ended.")
