import os
import openai
import logging
import re
import aiofiles

logger = logging.getLogger(__name__)

class Summarizer:
    """
    Transcript summarizer and transcript optimizer using OpenAI.
    Provides transcript optimization (cleanup, paragraphing) and summarization
    with chunking strategies for long texts.
    """

    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")

        if not api_key:
            logger.warning("OPENAI_API_KEY is not set. Summary features will be unavailable.")

        if api_key:
            if base_url:
                self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
                logger.info(f"OpenAI client initialized with custom base URL: {base_url}")
            else:
                self.client = openai.OpenAI(api_key=api_key)
                logger.info("OpenAI client initialized with default endpoint")
        else:
            self.client = None

        # language mapping for display
        self.language_map = {
            "en": "English",
            "zh": "Chinese (Simplified)",
            "es": "Español",
            "fr": "Français",
            "de": "Deutsch",
            "it": "Italiano",
            "pt": "Português",
            "ru": "Русский",
            "ja": "日本語",
            "ko": "한국어",
            "ar": "العربية"
        }

    async def optimize_transcript(self, raw_transcript: str) -> str:
        """
        Optimize raw transcript: remove timestamps/meta, fix typos, merge splitted sentences,
        and organize into readable paragraphs. Automatically chunk long transcripts.
        """
        try:
            if not self.client:
                logger.warning("OpenAI API unavailable — returning raw transcript.")
                return raw_transcript

            preprocessed = self._remove_timestamps_and_meta(raw_transcript)
            detected_lang_code = self._detect_transcript_language(preprocessed)
            max_chars_per_chunk = 4000

            if len(preprocessed) > max_chars_per_chunk:
                logger.info(f"Long transcript ({len(preprocessed)} chars). Using chunked optimization.")
                return await self._format_long_transcript_in_chunks(preprocessed, detected_lang_code, max_chars_per_chunk)
            else:
                return await self._format_single_chunk(preprocessed, detected_lang_code)

        except Exception as e:
            logger.error(f"Transcript optimization failed: {e}")
            logger.info("Returning raw transcript as fallback.")
            return raw_transcript

    def _estimate_tokens(self, text: str) -> int:
        """
        Conservative token estimation for chunking decisions.
        Combines crude character/word heuristics and overhead estimates.
        """
        chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        english_words = len([w for w in text.split() if w.isascii() and w.isalpha()])

        base_tokens = chinese_chars * 1.5 + english_words * 1.3
        format_overhead = len(text) * 0.15
        system_prompt_overhead = 2500

        total = int(base_tokens + format_overhead + system_prompt_overhead)
        return total

    async def _format_single_chunk(self, chunk_text: str, transcript_language: str = 'zh') -> str:
        """
        Format and optimize a single chunk. Uses Chat completions for fluency and paragraphing.
        """
        if not self.client:
            return self._apply_basic_formatting(chunk_text)

        lang_instruction = self._get_language_instruction(transcript_language)

        if transcript_language == 'zh':
            system_prompt = (
                "You are a professional transcript editing assistant. Fix errors and improve readability "
                "without changing meaning. Remove timestamps/metadata only. Do NOT change pronouns or speaker perspective. "
                "This could be an interview: interviewer uses 'you', interviewee uses 'I' or 'we'."
            )
            user_prompt = (
                "Optimize the following transcript text in Chinese. Remove timestamps, fix obvious errors, "
                "merge sentence fragments split by timestamps, and organize into natural paragraphs (1-8 related sentences per paragraph, "
                "paragraph length not exceed 400 characters). Keep original meaning and pronouns."
                f"\n\nOriginal Transcript:\n{chunk_text}"
            )
        else:
            system_prompt = (
                "You are a professional transcript editing assistant. Fix errors and improve readability "
                "without changing meaning. Remove timestamps/metadata only. DO NOT change pronouns or speaker perspective."
            )
            user_prompt = (
                f"Optimize the following transcript text in {lang_instruction}. Remove timestamps, fix obvious errors, "
                "merge sentence fragments split by timestamps, and organize into natural paragraphs (1-8 related sentences per paragraph, "
                "paragraph length not exceed 400 characters). Keep original meaning and pronouns."
                f"\n\nOriginal Transcript:\n{chunk_text}"
            )

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.1
            )
            optimized_text = response.choices[0].message.content or ""
            optimized_text = self._remove_transcript_heading(optimized_text)
            enforced = self._enforce_paragraph_max_chars(optimized_text.strip(), max_chars=400)
            return self._ensure_markdown_paragraphs(enforced)
        except Exception as e:
            logger.error(f"Single chunk formatting failed: {e}")
            return self._apply_basic_formatting(chunk_text)

    async def _format_long_transcript_in_chunks(self, raw_transcript: str, transcript_language: str, max_chars_per_chunk: int) -> str:
        """
        Chunk the transcript, optimize each chunk with context, deduplicate overlaps,
        and re-organize paragraphs at the end.
        """
        # split into sentence-like units
        parts = re.split(r"([。！？\.!?]+\s*)", raw_transcript)
        sentences = []
        buf = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                buf += part
            else:
                buf += part
                if buf.strip():
                    sentences.append(buf.strip())
                    buf = ""
        if buf.strip():
            sentences.append(buf.strip())

        # assemble chunks by size
        chunks = []
        cur = ""
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = s
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        # further split overly long chunks safely
        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                final_chunks.extend(self._smart_split_long_chunk(c, max_chars_per_chunk))

        logger.info(f"Transcript split into {len(final_chunks)} chunks for processing.")

        optimized = []
        for i, c in enumerate(final_chunks):
            chunk_with_context = c
            if i > 0:
                prev_tail = final_chunks[i - 1][-100:]
                marker = f"[Context continued: {prev_tail}]"
                chunk_with_context = marker + "\n\n" + c
            try:
                oc = await self._format_single_chunk(chunk_with_context, transcript_language)
                oc = re.sub(r"^\[Context continued:.*?\]\s*", "", oc, flags=re.S)
                optimized.append(oc)
            except Exception as e:
                logger.warning(f"Chunk {i+1} optimization failed, using basic formatting: {e}")
                optimized.append(self._apply_basic_formatting(c))

        # dedupe overlaps
        deduped = []
        for i, c in enumerate(optimized):
            cur_txt = c
            if i > 0 and deduped:
                prev = deduped[-1]
                overlap = self._find_overlap_between_texts(prev[-200:], cur_txt[:200])
                if overlap:
                    cur_txt = cur_txt[len(overlap):].lstrip()
                    if not cur_txt:
                        continue
            if cur_txt.strip():
                deduped.append(cur_txt)

        merged = "\n\n".join(deduped)
        merged = self._remove_transcript_heading(merged)
        enforced = self._enforce_paragraph_max_chars(merged, max_chars=400)
        return self._ensure_markdown_paragraphs(enforced)

    def _smart_split_long_chunk(self, text: str, max_chars_per_chunk: int) -> list:
        """
        Safely split extremely long blocks on sentence or space boundaries.
        """
        chunks = []
        pos = 0
        while pos < len(text):
            end = min(pos + max_chars_per_chunk, len(text))
            if end < len(text):
                sentence_endings = ['。', '！', '？', '.', '!', '?']
                best = -1
                for ch in sentence_endings:
                    idx = text.rfind(ch, pos, end)
                    if idx > best:
                        best = idx
                if best > pos + int(max_chars_per_chunk * 0.7):
                    end = best + 1
                else:
                    space_idx = text.rfind(' ', pos, end)
                    if space_idx > pos + int(max_chars_per_chunk * 0.8):
                        end = space_idx
            chunks.append(text[pos:end].strip())
            pos = end
        return [c for c in chunks if c]

    def _find_overlap_between_texts(self, text1: str, text2: str) -> str:
        """
        Find overlapping substring between suffix of text1 and prefix of text2 for deduplication.
        """
        max_len = min(len(text1), len(text2))
        for length in range(max_len, 19, -1):
            suffix = text1[-length:]
            prefix = text2[:length]
            if suffix == prefix:
                cut = self._find_safe_cut_point(prefix)
                if cut > 20:
                    return prefix[:cut]
                return suffix
        return ""

    def _apply_basic_formatting(self, text: str) -> str:
        """
        Basic fallback formatting: split into sentences and group into paragraphs.
        """
        if not text or not text.strip():
            return text
        parts = re.split(r"([。！？\.!?]+\s*)", text)
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:
                current += part
            else:
                current += part
                if current.strip():
                    sentences.append(current.strip())
                    current = ""
        if current.strip():
            sentences.append(current.strip())
        paras = []
        cur = ""
        sentence_count = 0
        for s in sentences:
            candidate = (cur + " " + s).strip() if cur else s
            sentence_count += 1
            should_break = False
            if len(candidate) > 400 and cur:
                should_break = True
            elif len(candidate) > 200 and sentence_count >= 3:
                should_break = True
            elif sentence_count >= 6:
                should_break = True

            if should_break:
                paras.append(cur.strip())
                cur = s
                sentence_count = 1
            else:
                cur = candidate
        if cur.strip():
            paras.append(cur.strip())
        return self._ensure_markdown_paragraphs("\n\n".join(paras))

    def _remove_timestamps_and_meta(self, text: str) -> str:
        """
        Remove timestamp lines and obvious metadata, keep spoken words and repetitions.
        """
        lines = text.split('\n')
        kept = []
        for line in lines:
            s = line.strip()
            if (s.startswith('**[') and s.endswith(']**')):
                continue
            if s.startswith('# '):
                continue
            if s.startswith('**Detected language:**') or s.startswith('**Language probability:**'):
                continue
            kept.append(line)
        cleaned = '\n'.join(kept)
        return cleaned

    def _enforce_paragraph_max_chars(self, text: str, max_chars: int = 400) -> str:
        """
        Ensure each paragraph does not exceed max_chars by splitting at sentence boundaries when needed.
        """
        if not text:
            return text
        paragraphs = [p for p in re.split(r"\n\s*\n", text) if p is not None]
        new_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if len(para) <= max_chars:
                new_paragraphs.append(para)
                continue
            parts = re.split(r"([。！？\.!?]+\s*)", para)
            sentences = []
            buf = ""
            for i, part in enumerate(parts):
                if i % 2 == 0:
                    buf += part
                else:
                    buf += part
                    if buf.strip():
                        sentences.append(buf.strip())
                        buf = ""
            if buf.strip():
                sentences.append(buf.strip())
            cur = ""
            for s in sentences:
                candidate = (cur + (" " if cur else "") + s).strip()
                if len(candidate) > max_chars and cur:
                    new_paragraphs.append(cur)
                    cur = s
                else:
                    cur = candidate
            if cur:
                new_paragraphs.append(cur)
        return "\n\n".join([p.strip() for p in new_paragraphs if p is not None])

    def _remove_transcript_heading(self, text: str) -> str:
        """
        Remove heading lines like '# Transcript' that may appear.
        """
        if not text:
            return text
        lines = text.split('\n')
        filtered = []
        for line in lines:
            stripped = line.strip()
            if re.match(r"^#{1,6}\s*transcript(\s+text)?\s*$", stripped, flags=re.I):
                continue
            filtered.append(line)
        return '\n'.join(filtered)

    def _split_into_chunks(self, text: str, max_tokens: int) -> list:
        """
        Split the raw transcript into chunks based on estimated token limits.
        """
        pure_text = self._extract_pure_text(text)
        sentences = self._split_into_sentences(pure_text)

        chunks = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = self._estimate_tokens(sentence)
            if current_tokens + sentence_tokens > max_tokens and current_chunk:
                chunks.append(self._join_sentences(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks.append(self._join_sentences(current_chunk))
        return chunks

    def _extract_pure_text(self, raw_transcript: str) -> str:
        """
        Extract plain text from raw transcript removing timestamps and metadata.
        """
        lines = raw_transcript.split('\n')
        text_lines = []
        for line in lines:
            line = line.strip()
            if (line.startswith('**[') and line.endswith(']**')) or line.startswith('#') or line.startswith('**Detected language:') or not line:
                continue
            text_lines.append(line)
        return ' '.join(text_lines)

    def _split_into_sentences(self, text: str) -> list:
        """
        Split text into sentence units, considering multilingual punctuation.
        """
        sentence_endings = r'[.!?。！？;；]+'
        parts = re.split(f'({sentence_endings})', text)
        sentences = []
        current = ""
        for i, part in enumerate(parts):
            if re.match(sentence_endings, part):
                current += part
                if current.strip():
                    sentences.append(current.strip())
                current = ""
            else:
                current += part
        if current.strip():
            sentences.append(current.strip())
        return [s for s in sentences if s.strip()]

    def _join_sentences(self, sentences: list) -> str:
        return ' '.join(sentences)

    def _basic_transcript_cleanup(self, raw_transcript: str) -> str:
        """
        A simple cleanup fallback removing timestamps, headings and metadata, then basic paragraphing.
        """
        lines = raw_transcript.split('\n')
        cleaned_lines = []
        for line in lines:
            if line.strip().startswith('**[') and line.strip().endswith(']**'):
                continue
            if line.strip().startswith('# ') or line.strip().startswith('## '):
                continue
            if line.strip().startswith('**Detected language:') or line.strip().startswith('**Language probability:'):
                continue
            if line.strip():
                cleaned_lines.append(line.strip())

        text = ' '.join(cleaned_lines)
        sentences = re.split(r'[.!?。！？]', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        paragraphs = []
        current_paragraph = []
        for i, sentence in enumerate(sentences):
            if sentence:
                current_paragraph.append(sentence)
                topic_keywords = [
                    'first', 'second', 'then', 'next', 'also', 'however', 'finally',
                    'now', 'so', 'therefore', 'but'
                ]
                should_break = False
                if len(current_paragraph) >= 3:
                    should_break = True
                elif len(current_paragraph) >= 2:
                    for kw in topic_keywords:
                        if sentence.lower().startswith(kw):
                            should_break = True
                            break
                if should_break or len(current_paragraph) >= 4:
                    paragraph_text = '. '.join(current_paragraph)
                    if not paragraph_text.endswith('.'):
                        paragraph_text += '.'
                    paragraphs.append(paragraph_text)
                    current_paragraph = []

        if current_paragraph:
            paragraph_text = '. '.join(current_paragraph)
            if not paragraph_text.endswith('.'):
                paragraph_text += '.'
            paragraphs.append(paragraph_text)

        return '\n\n'.join(paragraphs)

    async def _final_paragraph_organization(self, text: str, lang_instruction: str) -> str:
        """
        Final paragraph re-organization using the model when possible.
        """
        try:
            estimated_tokens = self._estimate_tokens(text)
            if estimated_tokens > 3000:
                return await self._organize_long_text_paragraphs(text, lang_instruction)

            system_prompt = (
                f"You are a professional paragraph organizer for {lang_instruction}. Reorganize paragraphs by semantics, "
                "ensuring each paragraph is a single coherent idea and <= 250 words. Keep original language and content."
            )
            user_prompt = f"Re-organize the following text into clear paragraphs in {lang_instruction}:\n\n{text}"

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4000,
                temperature=0.05
            )
            organized_text = response.choices[0].message.content
            validated_text = self._validate_paragraph_lengths(organized_text)
            return validated_text
        except Exception as e:
            logger.error(f"Final paragraph organization failed: {e}")
            return self._basic_paragraph_fallback(text)

    async def _organize_long_text_paragraphs(self, text: str, lang_instruction: str) -> str:
        """
        Organize very long text by chunking paragraphs and processing each chunk.
        """
        try:
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            organized_chunks = []
            current_chunk = []
            current_tokens = 0
            max_chunk_tokens = 2500

            for para in paragraphs:
                para_tokens = self._estimate_tokens(para)
                if current_tokens + para_tokens > max_chunk_tokens and current_chunk:
                    chunk_text = '\n\n'.join(current_chunk)
                    organized_chunk = await self._organize_single_chunk(chunk_text, lang_instruction)
                    organized_chunks.append(organized_chunk)
                    current_chunk = [para]
                    current_tokens = para_tokens
                else:
                    current_chunk.append(para)
                    current_tokens += para_tokens

            if current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                organized_chunk = await self._organize_single_chunk(chunk_text, lang_instruction)
                organized_chunks.append(organized_chunk)

            return '\n\n'.join(organized_chunks)
        except Exception as e:
            logger.error(f"Long-text paragraph organization failed: {e}")
            return self._basic_paragraph_fallback(text)

    async def _organize_single_chunk(self, text: str, lang_instruction: str) -> str:
        """
        Organize a single chunk's paragraphs.
        """
        system_prompt = (
            f"You are a {lang_instruction} paragraph organization expert. Reorganize paragraphs by semantics, "
            "ensuring each paragraph does not exceed 200 words and the content is complete."
        )
        user_prompt = f"Re-paragraph the following text in {lang_instruction}:\n\n{text}"

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1200,
            temperature=0.05
        )
        return response.choices[0].message.content

    def _validate_paragraph_lengths(self, text: str) -> str:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        validated = []
        for para in paragraphs:
            word_count = len(para.split())
            if word_count > 300:
                logger.warning(f"Detected oversized paragraph ({word_count} words), attempting to split.")
                split_paras = self._split_long_paragraph(para)
                validated.extend(split_paras)
            else:
                validated.append(para)
        return '\n\n'.join(validated)

    def _split_long_paragraph(self, paragraph: str) -> list:
        sentences = re.split(r'[.!?。！？]\s+', paragraph)
        sentences = [s.strip() + '.' for s in sentences if s.strip()]
        split_paragraphs = []
        cur = []
        cur_words = 0
        for sentence in sentences:
            sw = len(sentence.split())
            if cur_words + sw > 200 and cur:
                split_paragraphs.append(' '.join(cur))
                cur = [sentence]
                cur_words = sw
            else:
                cur.append(sentence)
                cur_words += sw
        if cur:
            split_paragraphs.append(' '.join(cur))
        return split_paragraphs

    def _basic_paragraph_fallback(self, text: str) -> str:
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        basic_paragraphs = []
        for para in paragraphs:
            word_count = len(para.split())
            if word_count > 250:
                split_paras = self._split_long_paragraph(para)
                basic_paragraphs.extend(split_paras)
            elif word_count < 30 and basic_paragraphs:
                last = basic_paragraphs[-1]
                combined_words = len(last.split()) + word_count
                if combined_words <= 200:
                    basic_paragraphs[-1] = last + ' ' + para
                else:
                    basic_paragraphs.append(para)
            else:
                basic_paragraphs.append(para)
        return '\n\n'.join(basic_paragraphs)

    async def summarize(self, transcript: str, target_language: str = "zh", video_title: str = None) -> str:
        """
        Generate a summary for the transcript in the target language. Uses chunking for long texts.
        """
        try:
            if not self.client:
                logger.warning("OpenAI API unavailable — generating fallback summary.")
                return self._generate_fallback_summary(transcript, target_language, video_title)

            estimated_tokens = self._estimate_tokens(transcript)
            max_summarize_tokens = 4000

            if estimated_tokens <= max_summarize_tokens:
                return await self._summarize_single_text(transcript, target_language, video_title)
            else:
                logger.info(f"Long transcript ({estimated_tokens} tokens). Using chunked summarization.")
                return await self._summarize_with_chunks(transcript, target_language, video_title, max_summarize_tokens)
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return self._generate_fallback_summary(transcript, target_language, video_title)

    async def _summarize_single_text(self, transcript: str, target_language: str, video_title: str = None) -> str:
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        system_prompt = (
            f"You are a professional content analyst. Please generate a comprehensive, well-structured summary in {language_name}."
        )
        user_prompt = (
            f"Write a comprehensive and well-structured summary in {language_name} for the following text:\n\n{transcript}\n\n"
            "Use natural paragraphs separated by double newlines. Keep a logical flow and preserve key points."
        )

        logger.info(f"Generating summary in {language_name}...")
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=3500,
            temperature=0.3
        )
        summary = response.choices[0].message.content
        return self._format_summary_with_meta(summary, target_language, video_title)

    async def _summarize_with_chunks(self, transcript: str, target_language: str, video_title: str, max_tokens: int) -> str:
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        chunks = self._smart_chunk_text(transcript, max_chars_per_chunk=4000)
        logger.info(f"Split transcript into {len(chunks)} chunks for summarization.")

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Summarizing chunk {i+1}/{len(chunks)}...")
            system_prompt = (
                f"You are a summarization expert. Create a high-density summary for this text chunk in {language_name}."
            )
            user_prompt = f"[Part {i+1}/{len(chunks)}] Summarize the key points of the following text in {language_name}:\n\n{chunk}"

            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=1000,
                    temperature=0.3
                )
                chunk_summary = response.choices[0].message.content
                chunk_summaries.append(chunk_summary)
            except Exception as e:
                logger.error(f"Chunk summarization failed for part {i+1}: {e}")
                simple_summary = f"Part {i+1} summary (fallback): {chunk[:200]}..."
                chunk_summaries.append(simple_summary)

        combined = "\n\n".join([f"[Part {idx+1}]\n{s}" for idx, s in enumerate(chunk_summaries)])
        logger.info("Integrating chunk summaries...")
        if len(chunk_summaries) > 10:
            final_summary = await self._integrate_hierarchical_summaries(chunk_summaries, target_language)
        else:
            final_summary = await self._integrate_chunk_summaries(combined, target_language)

        return self._format_summary_with_meta(final_summary, target_language, video_title)

    def _smart_chunk_text(self, text: str, max_chars_per_chunk: int = 3500) -> list:
        paragraphs = [p for p in text.split('\n\n') if p.strip()]
        cur = ""
        chunks = []
        for p in paragraphs:
            candidate = (cur + "\n\n" + p).strip() if cur else p
            if len(candidate) > max_chars_per_chunk and cur:
                chunks.append(cur.strip())
                cur = p
            else:
                cur = candidate
        if cur.strip():
            chunks.append(cur.strip())

        final_chunks = []
        for c in chunks:
            if len(c) <= max_chars_per_chunk:
                final_chunks.append(c)
            else:
                sentences = [s.strip() for s in re.split(r"[。！？\.!?]+", c) if s.strip()]
                scur = ""
                for s in sentences:
                    candidate = (scur + '。' + s).strip() if scur else s
                    if len(candidate) > max_chars_per_chunk and scur:
                        final_chunks.append(scur.strip())
                        scur = s
                    else:
                        scur = candidate
                if scur.strip():
                    final_chunks.append(scur.strip())
        return final_chunks

    async def _integrate_chunk_summaries(self, combined_summaries: str, target_language: str) -> str:
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        try:
            system_prompt = (
                f"You are a content integration expert. Integrate segmented summaries into a single coherent summary in {language_name}."
            )
            user_prompt = f"Integrate the following segmented summaries into a single coherent summary in {language_name}:\n\n{combined_summaries}"
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=2500,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Integrating chunk summaries failed: {e}")
            return combined_summaries

    def _format_summary_with_meta(self, summary: str, target_language: str, video_title: str = None) -> str:
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        prefix = f"# {video_title}\n\n" if video_title else ""
        return prefix + summary

    def _generate_fallback_summary(self, transcript: str, target_language: str, video_title: str = None) -> str:
        language_name = self.language_map.get(target_language, "Chinese (Simplified)")
        lines = transcript.split('\n')
        content_lines = [ln for ln in lines if ln.strip() and not ln.startswith('#') and not ln.startswith('**')]
        total_chars = sum(len(ln) for ln in content_lines)
        title = video_title if video_title else "Summary"

        summary = f"""# {title}

**Language:** {language_name}
**Notice:** OpenAI API is unavailable; this is a simplified summary.

## Transcript Overview

**Content length:** ~{total_chars} characters
**Paragraph count:** {len(content_lines)}

## Main Content

A simplified fallback summary is provided because the AI summarizer is not available.

## Suggestions

1. Review the full transcript manually.
2. Focus on paragraphs with timestamps for key points.
3. Manually extract major takeaways.

"""
        return summary

    def _get_current_time(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_supported_languages(self) -> dict:
        return self.language_map.copy()

    def _detect_transcript_language(self, transcript: str) -> str:
        """
        Detect primary language by looking for explicit markers or by simple char ratio heuristics.
        """
        if "**Detected language:**" in transcript:
            lines = transcript.split('\n')
            for line in lines:
                if "**Detected language:**" in line:
                    lang = line.split(":")[-1].strip()
                    return lang

        total_chars = len(transcript)
        if total_chars == 0:
            return "en"

        chinese_chars = sum(1 for ch in transcript if '\u4e00' <= ch <= '\u9fff')
        chinese_ratio = chinese_chars / total_chars
        english_chars = sum(1 for ch in transcript if ch.isascii() and ch.isalpha())
        english_ratio = english_chars / total_chars

        if chinese_ratio > 0.3:
            return "zh"
        elif english_ratio > 0.3:
            return "en"
        else:
            return "en"

    def _get_language_instruction(self, lang_code: str) -> str:
        mapping = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "es": "Spanish",
            "fr": "French",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "ru": "Russian",
            "ar": "Arabic"
        }
        return mapping.get(lang_code, "English")
