import numpy as np
import threading
import librosa

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

language_codes = ["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh","ja","hu","ko"]