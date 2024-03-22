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
python3 stream.py --model_path "./XTTS-v2/" --speaker_wavs "voice.wav" --language "en" --output_device 0 --model_temperature 0.65
```

If you are unsure of your output device index you can run list_devices.py to see all available outputs.

If you have DeepSpeed installed use
```bash
--deepspeed True
```