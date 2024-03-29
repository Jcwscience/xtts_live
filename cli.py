#!/usr/bin/env python

import argparse  # Import the argparse library
import sounddevice as sd
import os
from xtts_live.main import TextToSpeechGenerator

# Set up the argument parser
parser = argparse.ArgumentParser(description="TTS Generator with Command Line Arguments")
parser.add_argument("--model_path", type=str, default="./XTTS-v2/", help="Path to the TTS model directory")
parser.add_argument("--speaker_wavs", type=str, help="List of speaker WAV files for conditioning")
parser.add_argument("--language", type=str, default="en", help="Language for the TTS model")
parser.add_argument("--output_device", type=int, help="Output device ID for sound playback")
parser.add_argument("--model_temperature", type=float, default=0.65, help="Temperature parameter for the TTS model")
parser.add_argument("--list_devices", action="store_true", help="List available audio devices")
parser.add_argument("--use_deepspeed", type=bool, default=False, help="Use DeepSpeed for faster inference")
parser.add_argument("--cpu_only", type=bool, default=False, help="Use CPU for inference")
parser.add_argument("--enable_text_splitting", type=bool, default=True, help="Use text splitting for long texts")
parser.add_argument("--output_samplerate", type=int, default=48000, help="Output samplerate for the audio stream")


# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
if args.list_devices:
    print(sd.query_devices())
    exit(0)

model_path = args.model_path
if not os.path.exists(model_path):
    print("Error: Please provide a valid model path with --model_path.")
    exit(1)

speaker_wavs = args.speaker_wavs
output_device = args.output_device
output_samplerate = args.output_samplerate
enable_text_splitting = args.enable_text_splitting
use_deepspeed = args.use_deepspeed
language = args.language
cpu_only = args.cpu_only
model_temperature = args.model_temperature

# Initialize the synthesizer
tts = TextToSpeechGenerator(model_path, speaker_wavs, output_device, use_deepspeed, cpu_only, output_samplerate)

tts.start()

try:
    while True:
        input_text = input("Enter text: ")
        tts.speak(input_text)

except KeyboardInterrupt:
    tts.stop()