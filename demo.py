import sounddevice as sd
import os
from xtts_live.main import TextToSpeechGenerator

model_path = "/home/john/Documents/XTTS-v2/"
speaker_wavs = "/home/john/Documents/Voices/voice.wav"


tts = TextToSpeechGenerator(model_path, speaker_wavs, output_device=3, use_deepspeed=False)

try:
    while True:
        input_text = input("Enter text: ")
        tts.speak(input_text)

except KeyboardInterrupt:
    tts.stop()