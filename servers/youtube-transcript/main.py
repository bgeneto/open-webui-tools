"""
YouTube Transcript LLM Tool.

Provides YouTube video transcripts in the requested language (if available).

Author: bgeneto
Date: 2025-05-04
Version: 1.0.1
Last Modified: 2025-05-04
"""

import logging
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, VideoUnavailable
import re

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("youtube-transcript-tool")

# FastAPI app
app = FastAPI(
    title="YouTube Transcript LLM Tool",
    description="Provides YouTube video transcripts in the requested language (if available)",
    version="1.0.1",
)

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
#   Pydantic Models
# =========================


class YouTubeTranscriptInput(BaseModel):
    video_id_or_url: str = Field(
        ...,
        description="The YouTube video ID, URL, or short URL.",
    )
    language: str = Field(
        default="en",
        description="The language of the transcript (default is English).",
    )


# =========================
#   YouTubeTranscriptService
# =========================


class YouTubeTranscriptService:
    """
    Service for retrieving YouTube video transcripts.
    """

    def __init__(self):
        pass

    async def get_transcript(self, video_id: str, language: str) -> list:
        """
        Retrieve the transcript of a YouTube video.
        Args:
            video_id (str): The YouTube video ID.
            language (str): The language of the transcript.
        Returns:
            list: The transcript data.
        Raises:
            Exception: If the transcript is not found or the video is unavailable.
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            transcript = transcript_list.find_transcript([language])
            return transcript.fetch()
        except NoTranscriptFound:
            logger.error(
                f"No transcript found for video {video_id} in language {language}."
            )
            raise HTTPException(
                status_code=404,
                detail=f"No transcript found for video {video_id} in language {language}.",
            )
        except VideoUnavailable:
            logger.error(f"Video {video_id} is unavailable.")
            raise HTTPException(
                status_code=404,
                detail=f"Video {video_id} is unavailable.",
            )
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal error retrieving transcript.",
            )

    def extract_video_id(self, video_id_or_url: str) -> str:
        """
        Extract the YouTube video ID from a URL or short URL.
        Args:
            video_id_or_url (str): The YouTube video ID, URL, or short URL.
        Returns:
            str: The extracted YouTube video ID.
        """
        # Regular expression to match YouTube video IDs
        patterns = [
            r"^https://www\.youtube\.com/watch\?v=([^&]+)",  # Standard URL
            r"^https://youtu\.be/([^&]+)",  # Short URL
            r"^([a-zA-Z0-9_-]{11})$",  # Video ID
        ]
        for pattern in patterns:
            match = re.match(pattern, video_id_or_url)
            if match:
                return match.group(1)
        raise HTTPException(
            status_code=400,
            detail="Invalid YouTube video ID or URL.",
        )


youtube_transcript_service = YouTubeTranscriptService()


@app.post(
    "/get_transcript",
    summary="Retrieve the transcript of a YouTube video.",
    operation_id="get_transcript",
)
async def get_transcript(input: YouTubeTranscriptInput):
    """
    Endpoint to retrieve the transcript of a YouTube video.
    """
    video_id = youtube_transcript_service.extract_video_id(input.video_id_or_url)
    transcript = await youtube_transcript_service.get_transcript(
        video_id, input.language
    )
    return {"success": True, "transcript": transcript}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"Internal server error: {str(exc)}"},
    )
