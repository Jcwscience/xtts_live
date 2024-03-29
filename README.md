## Requirements

- numpy
- librosa
- sounddevice
- TTS

## Getting The Model

If you do not already have the xtts_v2 model you will need to download it.
Follow the instructions at https://huggingface.co/coqui/XTTS-v2 and specify the path to it when running the script.

## Usage

1. Set the `model_path` variable to the path of your XTTS-v2 model.
2. Set the `speaker_wavs` variable to the path of your speaker WAV files.
3. Create an instance of `TextToSpeechGenerator` with the model path, speaker WAVs, and other optional parameters.
4. Use the `speak` method of the TTS instance to generate speech from text.

Here is a basic example:

```python
import sounddevice as sd
from xtts_live.main import TextToSpeechGenerator

model_path = "/path/to/your/model"
speaker_wavs = "/path/to/your/wavs"

tts = TextToSpeechGenerator(model_path, speaker_wavs, output_device=3, use_deepspeed=False)

try:
    while True:
        input_text = input("Enter text: ")
        tts.speak(input_text)
except KeyboardInterrupt:
    tts.stop()