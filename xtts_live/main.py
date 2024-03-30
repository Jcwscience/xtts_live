import numpy as np
import threading
import librosa
import sounddevice as sd
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from queue import Queue, Empty
import time

class AudioBuffer:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)
        self.lock = threading.Lock()  # To ensure thread-safe operations on the buffer

    def add_data(self, new_data):
        with self.lock:
            self.buffer = np.concatenate((self.buffer, new_data), axis=0)

    def get_samples(self, n_samples):
        with self.lock:
            if len(self.buffer) >= n_samples:
                samples = self.buffer[:n_samples].reshape(-1, 1)
                self.buffer = self.buffer[n_samples:]
            else:
                samples = np.concatenate((self.buffer, np.zeros((n_samples - len(self.buffer),), dtype=np.float32)), axis=0).reshape(-1, 1)
                self.buffer = np.array([], dtype=np.float32)
        return samples

    def is_empty(self):
        with self.lock:
            return len(self.buffer) == 0

class TextToSpeech:
    def __init__(self, model_path, speaker_wavs, output_device, low_latency=True, use_deepspeed=False, use_cuda=True, output_samplerate=48000):
        self.output_device = output_device
        self.samplerate = output_samplerate
        self.task_queue = Queue()
        self.audio_buffer = AudioBuffer()
        self.processing = False
        self.stream = None
        self.low_latency = low_latency
        if self.low_latency:
            self._start_stream()

        # Load the TTS model
        self.config = XttsConfig()
        self.config.load_json(model_path + "config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, use_deepspeed=use_deepspeed)
        if use_cuda:
            self.model.cuda()
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wavs)

        # Thread initialization
        self.process_thread = threading.Thread(target=self._process_tasks)

    def _process_tasks(self):
            self.processing = True
            if not self.stream:
                self._start_stream()
                time.sleep(1)  # Wait for the stream to start
            while self.processing:
                try:
                    task = self.task_queue.get(timeout=1)  # Adjust timeout as needed
                    self._speak(task['text'], task['language'])
                except Empty:
                    if self.low_latency:
                        self.processing = False
                    else:
                        if self.audio_buffer.is_empty():
                            self.processing = False
                            self.stream.close()
                            self.stream = None


    def _stream_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        outdata[:] = self.audio_buffer.get_samples(frames)

    def _start_stream(self):
        self.stream = sd.OutputStream(device=self.output_device, samplerate=self.samplerate, channels=1, dtype=np.float32, callback=self._stream_callback)
        self.stream.start()

    def _speak(self, text, language, speaker_wav_paths):

        if speaker_wav_paths:
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wav_paths)
        else:
            gpt_cond_latent=self.gpt_cond_latent
            speaker_embedding=self.speaker_embedding

        chunks = self.model.inference_stream(text=text, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, temperature=0.65, enable_text_splitting=True)
        for chunk in chunks:
            if hasattr(chunk, 'cpu'):
                chunk = chunk.cpu().numpy()
            chunk = chunk.astype(np.float32)
            chunk = librosa.resample(chunk, orig_sr=24000, target_sr=self.samplerate)
            self.audio_buffer.add_data(chunk)

    def speak(self, text, language="en", speaker_wav_paths=None):
        if not self.process_thread.is_alive():
            self.process_thread = threading.Thread(target=self._process_tasks)
            self.process_thread.start()

        self.task_queue.put({'text': text, 'language': language, 'speaker_wav_paths': speaker_wav_paths})
        
    def stop(self):
        self.processing = False
        if self.stream:
            self.stream.abort(ignore_errors=True)
        self.process_thread.join()
        print("Audio stream stopped.")
