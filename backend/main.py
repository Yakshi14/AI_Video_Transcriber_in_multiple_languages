from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import os
import tempfile
import asyncio
import logging
from pathlib import Path
from typing import Optional
import aiofiles
import uuid
import json
import re

from video_processor import VideoProcessor
from transcriber import Transcriber
from summarizer import Summarizer
from translator import Translator

# Configure logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="AI Video Transcriber", version="1.0.0")

# CORS middleware config
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent

# Mount static files
app.mount("/static", StaticFiles(directory=str(PROJECT_ROOT / "static")), name="static")

# Create temporary directory
TEMP_DIR = PROJECT_ROOT / "temp"
TEMP_DIR.mkdir(exist_ok=True)

# Initialize processors
video_processor = VideoProcessor()
transcriber = Transcriber()
summarizer = Summarizer()
translator = Translator()

# Task state persistence with file storage
import json
import threading

TASKS_FILE = TEMP_DIR / "tasks.json"
tasks_lock = threading.Lock()

def load_tasks():
    """Load task states."""
    try:
        if TASKS_FILE.exists():
            with open(TASKS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except:
        pass
    return {}

def save_tasks(tasks_data):
    """Save task states to file."""
    try:
        with tasks_lock:
            with open(TASKS_FILE, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Failed to save task state: {e}")

async def broadcast_task_update(task_id: str, task_data: dict):
    """Broadcast SSE updates to all connected clients."""
    logger.info(f"Broadcasting task update: {task_id}, status: {task_data.get('status')}, connections: {len(sse_connections.get(task_id, []))}")
    if task_id in sse_connections:
        connections_to_remove = []
        for queue in sse_connections[task_id]:
            try:
                await queue.put(json.dumps(task_data, ensure_ascii=False))
            except Exception as e:
                logger.warning(f"Failed to send message to queue: {e}")
                connections_to_remove.append(queue)
        
        for queue in connections_to_remove:
            sse_connections[task_id].remove(queue)
        
        if not sse_connections[task_id]:
            del sse_connections[task_id]

# Load task states on startup
tasks = load_tasks()

# Prevent duplicate URL processing
processing_urls = set()

# Track active async tasks
active_tasks = {}

# Store SSE connections
sse_connections = {}

def _sanitize_title_for_filename(title: str) -> str:
    """Sanitize video title for safe filename."""
    if not title:
        return "untitled"
    safe = re.sub(r"[^\w\-\s]", "", title)
    safe = re.sub(r"\s+", "_", safe).strip("._-")
    return safe[:80] or "untitled"

@app.get("/")
async def read_root():
    """Return homepage."""
    return FileResponse(str(PROJECT_ROOT / "static" / "index.html"))

@app.post("/api/process-video")
async def process_video(
    url: str = Form(...),
    summary_language: str = Form(default="en")
):
    """Start processing video and return a task ID."""
    try:
        # Prevent duplicate URL processing
        if url in processing_urls:
            for tid, task in tasks.items():
                if task.get("url") == url:
                    return {"task_id": tid, "message": "This video is already being processed. Please wait..."}

        task_id = str(uuid.uuid4())
        processing_urls.add(url)

        tasks[task_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Starting video processing...",
            "script": None,
            "summary": None,
            "error": None,
            "url": url
        }
        save_tasks(tasks)

        task = asyncio.create_task(process_video_task(task_id, url, summary_language))
        active_tasks[task_id] = task

        return {"task_id": task_id, "message": "Task created. Processing started..."}

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

async def process_video_task(task_id: str, url: str, summary_language: str):
    """Async processing workflow."""
    try:
        tasks[task_id].update({
            "status": "processing",
            "progress": 10,
            "message": "Downloading video..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        await asyncio.sleep(0.1)

        tasks[task_id].update({
            "progress": 15,
            "message": "Extracting video information..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        audio_path, video_title = await video_processor.download_and_convert(url, TEMP_DIR)

        tasks[task_id].update({
            "progress": 35,
            "message": "Video downloaded. Preparing transcription..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        tasks[task_id].update({
            "progress": 40,
            "message": "Transcribing audio..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        raw_script = await transcriber.transcribe(audio_path)

        # Save raw transcription
        try:
            short_id = task_id.replace("-", "")[:6]
            safe_title = _sanitize_title_for_filename(video_title)
            raw_md_filename = f"raw_{safe_title}_{short_id}.md"
            raw_md_path = TEMP_DIR / raw_md_filename

            with open(raw_md_path, "w", encoding="utf-8") as f:
                content_raw = (raw_script or "") + f"\n\nsource: {url}\n"
                f.write(content_raw)

            tasks[task_id].update({
                "raw_script_file": raw_md_filename
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

        except Exception as e:
            logger.error(f"Failed to save raw transcription: {e}")

        tasks[task_id].update({
            "progress": 55,
            "message": "Optimizing transcript..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        script = await summarizer.optimize_transcript(raw_script)
        script_with_title = f"# {video_title}\n\n{script}\n\nsource: {url}\n"

        detected_language = transcriber.get_detected_language(raw_script)
        logger.info(f"Detected language: {detected_language}, Summary language: {summary_language}")

        translation_content = None
        translation_filename = None
        translation_path = None

        if detected_language and translator.should_translate(detected_language, summary_language):

            tasks[task_id].update({
                "progress": 70,
                "message": "Generating translation..."
            })
            save_tasks(tasks)
            await broadcast_task_update(task_id, tasks[task_id])

            translation_content = await translator.translate_text(script, summary_language, detected_language)
            translation_with_title = f"# {video_title}\n\n{translation_content}\n\nsource: {url}\n"

            translation_filename = f"translation_{safe_title}_{short_id}.md"
            translation_path = TEMP_DIR / translation_filename
            async with aiofiles.open(translation_path, "w", encoding="utf-8") as f:
                await f.write(translation_with_title)

        tasks[task_id].update({
            "progress": 80,
            "message": "Generating summary..."
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        summary = await summarizer.summarize(script, summary_language, video_title)
        summary_with_source = summary + f"\n\nsource: {url}\n"

        script_filename = f"transcript_{task_id}.md"
        script_path = TEMP_DIR / script_filename

        async with aiofiles.open(script_path, "w", encoding="utf-8") as f:
            await f.write(script_with_title)

        new_script_filename = f"transcript_{safe_title}_{short_id}.md"
        new_script_path = TEMP_DIR / new_script_filename
        try:
            if script_path.exists():
                script_path.rename(new_script_path)
                script_path = new_script_path
        except:
            pass

        summary_filename = f"summary_{safe_title}_{short_id}.md"
        summary_path = TEMP_DIR / summary_filename
        async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
            await f.write(summary_with_source)

        task_result = {
            "status": "completed",
            "progress": 100,
            "message": "Processing completed!",
            "video_title": video_title,
            "script": script_with_title,
            "summary": summary_with_source,
            "script_path": str(script_path),
            "summary_path": str(summary_path),
            "short_id": short_id,
            "safe_title": safe_title,
            "detected_language": detected_language,
            "summary_language": summary_language
        }

        if translation_content and translation_path:
            task_result.update({
                "translation": translation_with_title,
                "translation_path": str(translation_path),
                "translation_filename": translation_filename
            })

        tasks[task_id].update(task_result)
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

        processing_urls.discard(url)

        if task_id in active_tasks:
            del active_tasks[task_id]

    except Exception as e:
        logger.error(f"Task {task_id} failed: {str(e)}")

        processing_urls.discard(url)

        if task_id in active_tasks:
            del active_tasks[task_id]

        tasks[task_id].update({
            "status": "error",
            "error": str(e),
            "message": f"Processing failed: {str(e)}"
        })
        save_tasks(tasks)
        await broadcast_task_update(task_id, tasks[task_id])

@app.get("/api/task-status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]

@app.get("/api/task-stream/{task_id}")
async def task_stream(task_id: str):
    """SSE task status stream."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        queue = asyncio.Queue()

        if task_id not in sse_connections:
            sse_connections[task_id] = []
        sse_connections[task_id].append(queue)

        try:
            current_task = tasks.get(task_id, {})
            yield f"data: {json.dumps(current_task, ensure_ascii=False)}\n\n"

            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {data}\n\n"

                    task_data = json.loads(data)
                    if task_data.get("status") in ["completed", "error"]:
                        break

                except asyncio.TimeoutError:
                    yield f"data: {json.dumps({'type': 'heartbeat'}, ensure_ascii=False)}\n\n"

        finally:
            if task_id in sse_connections and queue in sse_connections[task_id]:
                sse_connections[task_id].remove(queue)
                if not sse_connections[task_id]:
                    del sse_connections[task_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET",
            "Access-Control-Allow-Headers": "Cache-Control"
        }
    )

@app.get("/api/download/{filename}")
async def download_file(filename: str):
    """Download generated .md files."""
    try:
        if not filename.endswith('.md'):
            raise HTTPException(status_code=400, detail="Only .md files can be downloaded")

        if '..' in filename or '/' in filename or '\\' in filename:
            raise HTTPException(status_code=400, detail="Invalid filename format")

        file_path = TEMP_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")

        return FileResponse(
            file_path,
            filename=filename,
            media_type="text/markdown"
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise HTTPException(status_code=500, detail=f"Download failed: {str(e)}")

@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """Cancel and delete a task."""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    if task_id in active_tasks:
        task = active_tasks[task_id]
        if not task.done():
            task.cancel()
            logger.info(f"Task {task_id} canceled")
        del active_tasks[task_id]

    task_url = tasks[task_id].get("url")
    if task_url:
        processing_urls.discard(task_url)

    del tasks[task_id]
    return {"message": "Task deleted"}

@app.get("/api/tasks/active")
async def get_active_tasks():
    """Debug: get active tasks."""
    return {
        "active_tasks": len(active_tasks),
        "processing_urls": len(processing_urls),
        "task_ids": list(active_tasks.keys())
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
