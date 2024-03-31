from xtts_live.main import TextToSpeech

model_path = "/home/john/Documents/XTTS-v2/"
speaker_wavs = "/home/john/Documents/Voices/voice.wav"


tts = TextToSpeech(model_path, speaker_wavs, output_device=3, use_deepspeed=False, debug=True)

try:
    while True:
        input_text = input("Enter text: ")
        tts.speak(input_text)

except KeyboardInterrupt:
    tts.stop()