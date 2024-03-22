#!/usr/bin/env python

import argparse  # Import the argparse library
import sounddevice as sd
import os

# Set up the argument parser
parser = argparse.ArgumentParser(description="TTS Generator with Command Line Arguments")
parser.add_argument("--model_path", type=str, default="./XTTS-v2/", help="Path to the TTS model directory")
parser.add_argument("--speaker_wavs", nargs='+', help="List of speaker WAV files for conditioning")
parser.add_argument("--language", type=str, default="en", help="Language for the TTS model")
parser.add_argument("--output_device", type=int, help="Output device ID for sound playback")
parser.add_argument("--model_temperature", type=float, default=0.65, help="Temperature parameter for the TTS model")
parser.add_argument("--list_devices", action="store_true", help="List available audio devices")
parser.add_argument("--use_deepspeed", action="store_true", help="Use DeepSpeed for faster inference")
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
if not all(os.path.exists(wav) for wav in speaker_wavs):
    print("Error: Please provide valid speaker WAV files with --speaker_wavs.")
    exit(1)
output_device = args.output_device
output_samplerate = args.output_samplerate
enable_text_splitting = args.enable_text_splitting
use_deepspeed = args.use_deepspeed
language = args.language


import numpy as np
import librosa
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


# Load model
print("Loading model...")
config = XttsConfig()
config.load_json(model_path + "config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, use_deepspeed=use_deepspeed)
model.cuda()

class AudioBuffer:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.keep_stream_open = True  # Flag to determine behavior when buffer is empty

    def add_data(self, new_data):
        # Append new data to the buffer.
        self.buffer = np.concatenate((self.buffer, new_data), axis=0)

    def get_samples(self, n_samples):
        if len(self.buffer) > 0:
            if n_samples <= len(self.buffer):
                # Enough data available, return requested samples
                samples = self.buffer[:n_samples].reshape(-1, 1)
                self.buffer = self.buffer[n_samples:]
            else:
                # Not enough data, pad with zeros
                padding = np.zeros(n_samples - len(self.buffer), dtype=np.float32)
                samples = np.concatenate((self.buffer, padding), axis=0).reshape(-1, 1)
                self.buffer = np.array([], dtype=np.float32)  # Clear the buffer
            return samples
        elif self.keep_stream_open:
            # Buffer is empty, but we want to keep the stream open
            return np.zeros(n_samples, dtype=np.float32).reshape(-1, 1)
        else:
            # Buffer is empty and we want to close the stream
            return None
        

audio_buffer = AudioBuffer()

def stream_callback(outdata, frames, time, status):
    # Get samples from the buffer
    data = audio_buffer.get_samples(frames)
    if data is not None:
        outdata[:] = data
    else:
        # End stream
        sd.CallbackAbort

def generate_speech(text, language=args.language, gpt_cond_latent=None, speaker_embedding=None):
    chunks = model.inference_stream(text, language, gpt_cond_latent, speaker_embedding, temperature=args.model_temperature, enable_text_splitting=enable_text_splitting)
    audio_buffer.keep_stream_open = True
    for chunk in chunks:
        # If the chunk is a tensor, convert it to a NumPy array and ensure it's a float32 array
        if hasattr(chunk, 'cpu'):
            chunk = chunk.cpu().numpy()
        chunk = chunk.astype(np.float32)
        chunk = librosa.resample(chunk, orig_sr=24000, target_sr=output_samplerate)
        audio_buffer.add_data(chunk)
    audio_buffer.keep_stream_open = False

# Get conditioning latents
print("Getting conditioning latents...")
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wavs)

# Start audio stream
print("Starting audio stream...")
stream = sd.OutputStream(samplerate=output_samplerate, channels=1, dtype=np.float32, device=output_device, blocksize=2048, callback=stream_callback)
stream.start()

try:
    while True:
        input_text = input("Enter text: ")
        generate_speech(input_text, args.language, gpt_cond_latent, speaker_embedding)

except KeyboardInterrupt:
    print("Stopping audio stream...")
    stream.close()
