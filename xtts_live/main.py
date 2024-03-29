import numpy as np
import threading
import librosa
import sounddevice as sd
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
import queue

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

class TextToSpeechGenerator:
    def __init__(self, model_path, speaker_wavs, output_device, use_deepspeed=False, cpu_only=False, output_samplerate=48000, **kwargs):
        self.output_device = output_device
        self.samplerate = output_samplerate
        self.kwargs = kwargs
        self.task_queue = queue.Queue()
        self.audio_buffer = AudioBuffer()
        self.running = False
        self.thread = threading.Thread(target=self._process_queue)
        self.config = XttsConfig()
        self.config.load_json(model_path + "config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, use_deepspeed=use_deepspeed)
        if not cpu_only:
            self.model.cuda()
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wavs)


    def _process_queue(self):
        while self.running:
            task = self.task_queue.get()
            self._speak(task['text'], task['language'])

    def _stream_callback(self, outdata, frames, time, status):
        # Get samples from the buffer
        data = self.audio_buffer.get_samples(frames)
        if data is not None:
            outdata[:] = data
        else:
            # End stream
            sd.CallbackAbort

    def _speak(self, text, language):
        chunks = self.model.inference_stream(text=text, language=language, gpt_cond_latent=self.gpt_cond_latent, speaker_embedding=self.speaker_embedding, temperature=0.65, enable_text_splitting=True)
        self.audio_buffer.keep_stream_open = True
        for chunk in chunks:
            # If the chunk is a tensor, convert it to a NumPy array and ensure it's a float32 array
            if hasattr(chunk, 'cpu'):
                chunk = chunk.cpu().numpy()
            chunk = chunk.astype(np.float32)
            chunk = librosa.resample(chunk, orig_sr=24000, target_sr=self.samplerate)
            self.audio_buffer.add_data(chunk)
        self.audio_buffer.keep_stream_open = False

    def start(self):
        self.stream = sd.OutputStream(samplerate=self.samplerate, device=self.output_device, channels=1, dtype=np.float32, blocksize=2048, callback=self._stream_callback)
        self.stream.start()
        self.running = True
        self.thread.start()
        print("Audio stream started.")

    def speak(self, text, language="en"):
        self.task_queue.put({'text': text, 'language': language})

    def stop(self):
        self.running = False
        self.stream.close()
        print("Stopping audio stream...")
        self.thread.join()
        print("Audio stream stopped.")
        
        