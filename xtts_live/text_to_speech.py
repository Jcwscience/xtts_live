import numpy as np
import threading
import librosa
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from queue import Queue, Empty
import langid
import logging

# Stop deepstream from spewing out logs
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# Gag anything else we can think of while we're at it including the root logger
logging.disable(50)
logging.getLogger().setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)



class AudioBuffer:
    def __init__(self):
        self.buffer = np.array([], dtype=np.float32)

    def add_data(self, new_data):
        self.buffer = np.concatenate((self.buffer, new_data), axis=0)

    def get_samples(self, n_samples):
        if len(self.buffer) >= n_samples:
            samples = self.buffer[:n_samples]
            self.buffer = self.buffer[n_samples:]
        elif len(self.buffer) > 0:
            # Pad with zeros
            samples = np.concatenate((self.buffer, np.zeros(n_samples - len(self.buffer), dtype=np.float32)))
        else:
            samples = np.zeros(n_samples, dtype=np.float32)

        return samples.reshape(-1, 1)
    
    def clear(self):
        self.buffer = np.array([], dtype=np.float32)

    def is_empty(self):
        return len(self.buffer) == 0

class TextToSpeech:
    """
    A class that provides text-to-speech functionality using a pre-trained model.

    Args:
        model_path (str): The path to the directory containing the model files.
        speaker_wav_paths (str): The path to the directory containing the speaker's audio files.
        samplerate (int, optional): The desired sample rate of the generated audio. Defaults to 48000.
        use_deepspeed (bool, optional): Whether to use DeepSpeed for model loading. Defaults to False.
        use_cuda (bool, optional): Whether to use CUDA for model inference. Defaults to True.
        debug (bool, optional): Whether to enable debug mode. Defaults to False.

    Attributes:
        task_queue (Queue): A queue to store the text-to-speech tasks.
        audio_buffer (AudioBuffer): An audio buffer to store the generated audio.
        processing (bool): Whether the text-to-speech tasks are being processed.
        process_thread (Thread): The thread for processing the text-to-speech tasks.

    Methods:
        speak(text, language="en", speaker_wav_paths=None, temperature=0.65, enable_text_splitting=True):
            Adds a text-to-speech task to the queue for processing.
        stop():
            Stops the text-to-speech processing and clears the audio buffer.
    """
    
    def __init__(self, model_path, speaker_wav_paths, samplerate=48000, use_deepspeed=False, use_cuda=True, debug=False):
        # Initialize the TextToSpeech class
        self.model_path = model_path
        self.samplerate = samplerate
        self.use_deepspeed = use_deepspeed
        self.use_cuda = use_cuda
        self.debug = debug
        self.task_queue = Queue()
        self.audio_buffer = AudioBuffer()
        self.processing = False
        langid.set_languages(['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 'nl', 'cs', 'ar', 'zh', 'ja', 'hu', 'ko', 'hi'])

        # Load the model
        if self.debug: print("Loading model")
        self._load_model()
        self.gpt_cond_latent, self.speaker_embedding = self.model.get_conditioning_latents(audio_path=speaker_wav_paths)
        if self.debug: print("Model loaded")

        self.process_thread = threading.Thread(target=self._process_tasks)

    def _load_model(self):
        config = XttsConfig()
        config.load_json(self.model_path + "config.json")
        model = Xtts.init_from_config(config)
        model.load_checkpoint(config, checkpoint_dir=self.model_path, use_deepspeed=self.use_deepspeed)
        if self.use_cuda:
            model.cuda()
        self.model = model


    def _process_tasks(self):
            self.processing = True
            try:
                self.stream.start_stream()
            except: pass
            while self.processing:
                try:
                    task = self.task_queue.get(timeout=1)  # Adjust timeout as needed
                    self._speak(task['text'], task['language'], task['speaker_wav_paths'], task['temperature'], task['enable_text_splitting'])
                except Empty:
                    self.processing = False
        

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

    def speak(self, text, language=None, speaker_wav_paths=None, temperature=0.65, enable_text_splitting=True):
        if language is None:
                language, probability = langid.classify(text)
        """
        Adds a text-to-speech task to the queue.

        Args:
            text (str): The text to be converted to speech.
            language (str, optional): The language code for the text. Defaults to "en".
            speaker_wav_paths (list, optional): List of paths to speaker-specific wave files. Defaults to None.
            temperature (float, optional): The temperature parameter for controlling the randomness of the generated speech. Defaults to 0.65.
            enable_text_splitting (bool, optional): Flag to enable text splitting for long texts. Defaults to True.
        """
        if not self.process_thread.is_alive():
            if self.debug: print("Starting queue manager")
            self.process_thread = threading.Thread(target=self._process_tasks)
            self.process_thread.start()
        if self.debug: print("Adding task to queue")
        self.task_queue.put({'text': text, 'language': language, 'speaker_wav_paths': speaker_wav_paths, "temperature": temperature, "enable_text_splitting": enable_text_splitting})
        
    def stop(self):
        """
        Stops the queue manager and clears the audio buffer.

        This method sets the `processing` flag to False, puts a `None` value in the task queue to signal the end of processing,
        joins the process thread to wait for it to finish, and clears the audio buffer. If the `debug` flag is set, it also
        prints a message indicating that the queue manager has stopped.

        """
        if self.debug: 
            print("Stopping queue manager")
        self.processing = False
        self.task_queue.put(None)
        if self.process_thread.is_alive():
            self.process_thread.join()
        self.audio_buffer.clear()
        if self.debug: 
            print("Queue manager stopped")
