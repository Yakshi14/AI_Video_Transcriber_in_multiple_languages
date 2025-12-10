import os
import logging
import openai

logger = logging.getLogger(__name__)

class Translator:
    """
    Small translator wrapper using the OpenAI chat completions API.
    Provides should_translate() and translate_text().
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            logger.warning("OPENAI_API_KEY not set. Translator will not work.")
            self.client = None
        else:
            if base_url:
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                self.client = openai.OpenAI(api_key=api_key)

    def should_translate(self, detected_language: str, target_language: str) -> bool:
        """
        Decide whether translation is necessary. Returns True if both are present and different.
        """
        if not detected_language:
            return False
        if detected_language.lower() == target_language.lower():
            return False
        return True

    async def translate_text(self, text: str, target_language: str, source_language: str = None) -> str:
        """
        Translate text into the target language using OpenAI chat completions.
        """
        if not self.client:
            logger.warning("OpenAI API not configured; translation not available.")
            return text

        source_hint = source_language if source_language else "original language"
        system_prompt = (
            f"You are a professional translator. Translate the provided text into {target_language}. "
            f"Keep the original meaning, tone, and speaker perspective. Do not add or remove information."
        )
        user_prompt = f"Translate the following ({source_hint}) to {target_language}:\n\n{text}"

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=3000,
                temperature=0.1
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text
