import os
import yt_dlp
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processor for downloading and converting audio using yt-dlp."""

    def __init__(self):
        self.ydl_options = {
            "format": "bestaudio/best",  # Prefer best available audio
            "outtmpl": "%(title)s.%(ext)s",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    # Convert directly to mono 16k audio (smaller and stable)
                    "preferredcodec": "m4a",
                    "preferredquality": "192",
                }
            ],
            # Global FFmpeg args: mono + 16k sample rate + faststart
            "postprocessor_args": ["-ac", "1", "-ar", "16000", "-movflags", "+faststart"],
            "prefer_ffmpeg": True,
            "quiet": True,
            "no_warnings": True,
            "noplaylist": True,  # Force single video download, no playlist
        }

    async def download_and_convert(self, url: str, output_dir: Path) -> tuple[str, str]:
        """
        Download the video and extract audio as m4a.

        Args:
            url: Video URL
            output_dir: Directory to save the audio file

        Returns:
            A tuple containing:
            - Path to the converted audio file
            - Video title
        """
        try:
            # Ensure output directory exists
            output_dir.mkdir(exist_ok=True)

            # Generate unique filename prefix
            import uuid
            unique_id = str(uuid.uuid4())[:8]
            output_template = str(output_dir / f"audio_{unique_id}.%(ext)s")

            # Update yt-dlp options
            ydl_opts = self.ydl_options.copy()
            ydl_opts["outtmpl"] = output_template

            logger.info(f"Starting video download: {url}")

            import asyncio
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Fetch video metadata
                info = await asyncio.to_thread(ydl.extract_info, url, False)
                video_title = info.get("title", "unknown")
                expected_duration = info.get("duration") or 0
                logger.info(f"Video title: {video_title}")

                # Download audio
                await asyncio.to_thread(ydl.download, [url])

            # Determine final audio file path
            audio_file = str(output_dir / f"audio_{unique_id}.m4a")

            # If m4a is missing, check alternative formats
            if not os.path.exists(audio_file):
                for ext in ["webm", "mp4", "mp3", "wav"]:
                    candidate = str(output_dir / f"audio_{unique_id}.{ext}")
                    if os.path.exists(candidate):
                        audio_file = candidate
                        break
                else:
                    raise Exception("Audio file not found after download")

            # Validate duration and repair if necessary
            try:
                import subprocess, shlex

                probe_cmd = (
                    f"ffprobe -v error -show_entries format=duration "
                    f"-of default=noprint_wrappers=1:nokey=1 {shlex.quote(audio_file)}"
                )
                out = subprocess.check_output(probe_cmd, shell=True).decode().strip()
                actual_duration = float(out) if out else 0.0
            except Exception:
                actual_duration = 0.0

            if (
                expected_duration
                and actual_duration
                and abs(actual_duration - expected_duration) / expected_duration > 0.1
            ):
                logger.warning(
                    f"Audio duration mismatch: expected {expected_duration}s, "
                    f"got {actual_duration}s. Attempting repair..."
                )
                try:
                    fixed_path = str(output_dir / f"audio_{unique_id}_fixed.m4a")
                    fix_cmd = (
                        f"ffmpeg -y -i {shlex.quote(audio_file)} -vn -c:a aac -b:a 160k "
                        f"-movflags +faststart {shlex.quote(fixed_path)}"
                    )
                    subprocess.check_call(fix_cmd, shell=True)

                    audio_file = fixed_path
                    logger.info("Audio repair completed.")
                except Exception as e:
                    logger.error(f"Audio repair failed: {e}")

            logger.info(f"Audio file saved: {audio_file}")
            return audio_file, video_title

        except Exception as e:
            logger.error(f"Video download failed: {str(e)}")
            raise Exception(f"Video download failed: {str(e)}")

    def get_video_info(self, url: str) -> dict:
        """
        Get basic video metadata.

        Args:
            url: Video URL

        Returns:
            A dictionary containing video metadata.
        """
        try:
            with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
                info = ydl.extract_info(url, download=False)
                return {
                    "title": info.get("title", ""),
                    "duration": info.get("duration", 0),
                    "uploader": info.get("uploader", ""),
                    "upload_date": info.get("upload_date", ""),
                    "description": info.get("description", ""),
                    "view_count": info.get("view_count", 0),
                }
        except Exception as e:
            logger.error(f"Failed to fetch video info: {str(e)}")
            raise Exception(f"Failed to fetch video info: {str(e)}")
