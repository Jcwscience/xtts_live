from xtts_live import TextToSpeech
import sounddevice as sd


model_path = "/home/john/Documents/XTTS-v2/"
speaker_wavs = "/home/john/Documents/Voices/voice.wav"

output_samplerate = 48000

tts = TextToSpeech(model_path, speaker_wavs, output_samplerate, use_deepspeed=True)

def stream_callback(outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = tts.audio_buffer.get_samples(frames)

stream = sd.OutputStream(
    device=3,
    samplerate=output_samplerate,
    channels=1,
    callback=stream_callback,
    finished_callback=tts.stop,
    dtype='float32'
)

stream.start()

try:
    while True:
        input_text = input("Enter text: ")
        tts.speak(input_text)

except KeyboardInterrupt:
    tts.stop()
    stream.stop()