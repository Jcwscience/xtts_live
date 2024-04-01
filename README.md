# xtts_live

The aim of this project is to provide a simple wrapper around XTTS-v2 that allows for low latency streaming output.

## Requirements

- numpy
- librosa
- TTS
- An audio stream backend such as pyaudio or sounddevice

## Getting The Model

If you do not already have the xtts_v2 model you will need to download it.
Follow the instructions at https://huggingface.co/coqui/XTTS-v2 and specify the path to it when running the script.

## Usage

```python
# Import the wrapper
from xtts_live import TextToSpeech

# Initialize an instance of the TextToSpeech class
TTS = TextToSpeech(model_path, speaker_wavs)

# Add text to the processing queue
TTS.speak("Text to be spoken.")

# Read frames from the audio buffer
TTS.audio_buffer.get_samples("Number of samples to retrieve")

# Clean up the tts buffers and threads
TTS.stop()
```

See demo.py for example stream setup and integration.