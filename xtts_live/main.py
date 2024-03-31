import numpy as np
import threading
import librosa
import pyaudio
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
                samples = self.buffer.reshape(-1, 1)
        return samples

    def is_empty(self):
        with self.lock:
            return len(self.buffer) == 0

class TextToSpeech:
    def __init__(self, model_path, speaker_wav_paths, output_device, output_samplerate=48000, use_deepspeed=False, use_cuda=True, debug=False):
        self.debug = debug
        self.use_cuda = use_cuda
        self.speaker_wav_paths = speaker_wav_paths
        self.output_device = output_device
        self.samplerate = output_samplerate
        self.task_queue = Queue()
        self.audio_buffer = AudioBuffer()
        self.processing = False

        # Load the TTS model
        if self.debug:
            print("Loading TTS model...")
        self.config = XttsConfig()
        self.config.load_json(model_path + "config.json")
        self.model = Xtts.init_from_config(self.config)
        self.model.load_checkpoint(self.config, checkpoint_dir=model_path, use_deepspeed=use_deepspeed)
        if self.use_cuda:
            self.model.cuda()
        if self.debug:
            print("Primimg speaker latents...")
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=self.speaker_wav_paths)
        if debug:
            print("Opening audio stream...")
        self.stream = pyaudio.PyAudio()
        self.stream.open(output_device_index=self.output_device, channels=1 , format=pyaudio.paFloat32, output=True, rate=self.samplerate, frames_per_buffer=2048, start=False, stream_callback=self._stream_callback)
        # Thread initialization
        self.process_thread = threading.Thread(target=self._process_tasks)
        if self.debug:
            print("TextToSpeech initialized.")

    def _process_tasks(self):
            self.stream.start_stream()
            self.processing = True
            while self.processing:
                try:
                    task = self.task_queue.get(timeout=1)  # Adjust timeout as needed
                    self._speak(task['text'], task['language'], task['speaker_wav_paths'], task['temperature'], task['enable_text_splitting'])
                except Empty:
                    self.processing = False


    def _stream_callback(self, in_data, frame_count, time_info, status_flags):
        if self.audio_buffer.is_empty():
            if self.processing:
                return b'\0' * frame_count * 1, pyaudio.paPrimingOutput
            else:
                return None, pyaudio.paComplete
        else:
            samples = self.audio_buffer.get_samples(frame_count)
        return samples.tobytes(), pyaudio.paContinue

    def _speak(self, text, language, speaker_wav_paths, temperature, enable_text_splitting):

        if speaker_wav_paths:
            gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wav_paths)
        else:
            gpt_cond_latent=self.gpt_cond_latent
            speaker_embedding=self.speaker_embedding

        chunks = self.model.inference_stream(text=text, language=language, gpt_cond_latent=gpt_cond_latent, speaker_embedding=speaker_embedding, temperature=temperature, enable_text_splitting=enable_text_splitting)
        for chunk in chunks:
            if self.use_cuda:
                chunk = chunk.cpu().numpy()
            chunk = chunk.astype(np.float32)
            chunk = librosa.resample(chunk, orig_sr=24000, target_sr=self.samplerate)
            self.audio_buffer.add_data(chunk)

    def speak(self, text, language="en", speaker_wav_paths=None, temperature=0.65, enable_text_splitting=True):
        if not self.process_thread.is_alive():
            self.process_thread = threading.Thread(target=self._process_tasks)
            self.process_thread.start()

        self.task_queue.put({'text': text, 'language': language, 'speaker_wav_paths': speaker_wav_paths, "temperature": temperature, "enable_text_splitting": enable_text_splitting})
        
    def stop(self):
        self.processing = False
        self.stream.stop_stream()
        self.stream.close()
        self.stream.terminate()
        self.process_thread.join()
        if self.debug:
            print("Audio stream stopped.")
