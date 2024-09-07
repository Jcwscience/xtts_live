from xtts_live import TextToSpeech  # Import the TextToSpeech class from the xtts_live module
import sounddevice as sd  # Import the sounddevice module for audio streaming

model_path = "/home/john/.local/share/tts/tts_models--multilingual--multi-dataset--xtts_v2/"  # Path to the model directory
speaker_wavs = "/home/john/Documents/Voices/voice.wav"  # Path to the speaker's audio file

output_samplerate = 48000  # Output audio sample rate


tts = TextToSpeech(model_path, speaker_wavs, output_samplerate, use_deepspeed=True, debug=False)
# Create an instance of the TextToSpeech class with the specified model path, speaker audio path, output sample rate, and use_deepspeed flag

def stream_callback(outdata, frames, time, status):
    if status:
        print(status)  # Print any status messages received during audio streaming
    outdata[:] = tts.audio_buffer.get_samples(frames)  # Fill the output audio buffer with samples from the TextToSpeech instance


stream = sd.OutputStream(
    device=3,  # Specify the audio output device index
    samplerate=output_samplerate,  # Set the sample rate for audio streaming
    channels=1,  # Set the number of audio channels
    callback=stream_callback,  # Set the callback function for audio streaming
    finished_callback=tts.stop,  # Set the finished callback function to stop the TextToSpeech instance
    dtype='float32'  # Set the data type for audio samples
)

stream.start()  # Start the audio streaming

try:
    while True:
        input_text = input("Enter text: ")  # Prompt the user to enter text
        # If no language is specified, the language will be detected automatically
        # If the language cannot be detected, the default language "en" will be used
        # If something else goes wrong I have no idea what will happen
        tts.speak(input_text) # Add the input text to the text-to-speech queue for processing

except KeyboardInterrupt:
    tts.stop()  # Stop the TextToSpeech instance
    stream.stop()  # Stop the audio streaming
