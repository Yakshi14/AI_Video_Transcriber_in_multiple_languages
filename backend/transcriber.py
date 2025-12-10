import os
import logging

logger = logging.getLogger(__name__)

class Transcriber:
    """
    Transcriber class — tries to use faster_whisper if available; falls back to a very simple
    placeholder that returns empty text if unavailable. It also exposes a get_detected_language method.
    """

    def __init__(self):
        self.model = None
        self.model_name = os.getenv("WHISPER_MODEL_SIZE", "base")
        try:
            from faster_whisper import WhisperModel
            logger.info(f"Loading faster_whisper model: {self.model_name}")
            self.model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        except Exception as e:
            logger.warning(f"faster_whisper not available or failed to load: {e}. Transcription will be limited.")
            self.model = None

    async def transcribe(self, audio_path: str) -> str:
        """
        Transcribe an audio file and return the concatenated text transcript.
        """
        try:
            if self.model:
                segments, info = self.model.transcribe(audio_path, beam_size=5)
                texts = []
                for segment in segments:
                    # each segment has .text (string)
                    texts.append(segment.text)
                transcript = "\n".join(texts)
                # append detected language metadata if available
                if hasattr(info, "language"):
                    transcript = f"**Detected language:** {info.language}\n" + transcript
                return transcript
            else:
                # fallback: return empty or a simple message
                logger.warning("No ASR model available; returning empty transcript.")
                return ""
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return ""

    def get_detected_language(self, transcript: str) -> str:
        """
        Attempt to extract a detected language code from the transcript text (if inserted by transcribe),
        or fallback to heuristics.
        """
        if not transcript:
            return "en"
        # try marker
        for line in transcript.split("\n"):
            if line.startswith("**Detected language:**"):
                parts = line.split(":")
                if len(parts) >= 2:
                    return parts[-1].strip()
        # fallback heuristic: count Chinese chars
        total = len(transcript)
        if total == 0:
            return "en"
        chinese_chars = sum(1 for ch in transcript if '\u4e00' <= ch <= '\u9fff')
        if chinese_chars / total > 0.3:
            return "zh"
        # default to english
        return "en"
