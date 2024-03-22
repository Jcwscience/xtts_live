## Requirements

- numpy
- librosa
- sounddevice
- argparse
- TTS

## Getting The Model

If you do not already have the xtts_v2 model you will need to download it.
Follow the instructions at https://huggingface.co/coqui/XTTS-v2 and specify the path to it when running the script.

## Usage

You can run basic live inference from the command line. Here is an example:

```bash
./stream.py --model_path "./XTTS-v2/" --speaker_wavs "voice.wav" --output_device 0
```

### Aditional Options

```bash
--list_devices: List available audio devices
--language: Language for the TTS model, Default="en"
--enable_text_splitting: Use text splitting for long texts, Default=True
--output_samplerate: Output samplerate for the audio stream, Default=48000
--use_deepspeed: Use DeepSpeed for faster inference, Default=False
--model_temperature: Temperature parameter for the TTS model, Default=0.65
```
