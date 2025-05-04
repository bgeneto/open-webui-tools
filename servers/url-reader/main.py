"""
Simple Math Tools API.

Provides LLM tools for fetch web pages (URLs) and convert them to LLM-friendly formats like markdown, text.

Author: bgeneto
Date: 2025-05-02
Version: 1.0.2
Last Modified: 2025-05-02
"""

import traceback
from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware
import httpx
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("reader-tools")

app = FastAPI(
    title="URL Reader",
    description="AProvides LLM tools for fetch web pages (URLs) and convert them to LLM-friendly formats like markdown, text and others like screenshots.",
    version="1.0.2",
)

# Enable CORS (allowing all origins for simplicity)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simulated base URL for the Reader tool
READER_BASE_URL = "http://reader:3000"

# Supported response types
RESPONSE_TYPES = {
    "markdown": "markdown",
    "html": "html",
    "text": "text",
    "screenshot": "screenshot",
    "pageshot": "pageshot",
}


# Helper to fetch and wrap responses
async def _fetch_content(url: str, respond_with: str, client_params: dict = None):
    target_url = f"{READER_BASE_URL}/{url}"
    headers = {"X-Respond-With": respond_with}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                target_url, headers=headers, params=client_params, follow_redirects=True
            )
            response.raise_for_status()
            content_type = f"text/{respond_with}"
            if "application/json" in content_type:
                return JSONResponse(content=response.json())
            elif (
                "text/html" in content_type
                or "text/plain" in content_type
                or "text/markdown" in content_type
            ):
                return JSONResponse(
                    content={"content": response.text, "content_type": content_type}
                )
            else:
                import base64

                return JSONResponse(
                    content={
                        "content_base64": base64.b64encode(response.content).decode(
                            "utf-8"
                        ),
                        "content_type": "image/png",
                    }
                )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


# Markdown endpoint
@app.get(
    "/markdown/{url:path}",
    summary="Fetch markdown of a URL",
    operation_id="get_markdown",
    responses={
        200: {
            "description": "Markdown content in JSON",
            "content": {
                "application/json": {
                    "example": {"content": "# Heading", "content_type": "text/markdown"}
                }
            },
        }
    },
)
async def get_markdown(url: str):
    return await _fetch_content(url, "markdown")


# HTML endpoint
@app.get(
    "/html/{url:path}",
    summary="Fetch HTML of a URL",
    operation_id="get_html",
    responses={
        200: {
            "description": "HTML content in JSON",
            "content": {
                "application/json": {
                    "example": {
                        "content": "<html>...</html>",
                        "content_type": "text/html",
                    }
                }
            },
        }
    },
)
async def get_html(url: str):
    return await _fetch_content(url, "html")


# Text endpoint
@app.get(
    "/text/{url:path}",
    summary="Fetch plain text of a URL",
    operation_id="get_text",
    responses={
        200: {
            "description": "Text content in JSON",
            "content": {
                "application/json": {
                    "example": {
                        "content": "Plain text...",
                        "content_type": "text/plain",
                    }
                }
            },
        }
    },
)
async def get_text(url: str):
    return await _fetch_content(url, "plain")


# Screenshot endpoint with dimensions
@app.get(
    "/screenshot/{url:path}",
    summary="Fetch screenshot of a URL",
    operation_id="get_screenshot",
    responses={
        200: {
            "description": "Screenshot as base64 image data",
            "content": {
                "application/json": {
                    "example": {
                        "content_base64": "iVBORw0KGgoAAAANS...",
                        "content_type": "image/png",
                    }
                }
            },
        }
    },
)
async def get_screenshot(
    url: str,
    width: int = Query(1280, description="Viewport width"),
    height: int = Query(720, description="Viewport height"),
):
    import base64

    target_url = f"{READER_BASE_URL}/{url}"
    headers = {"X-Respond-With": "screenshot"}
    params = {"width": width, "height": height}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                target_url, headers=headers, params=params, follow_redirects=True
            )
            response.raise_for_status()
            content_base64 = base64.b64encode(response.content).decode("utf-8")
            return JSONResponse(
                content={"content_base64": content_base64, "content_type": "image/png"}
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


# Pageshot endpoint
@app.get(
    "/pageshot/{url:path}",
    summary="Fetch full page screenshot of a URL",
    operation_id="get_pageshot",
    responses={
        200: {
            "description": "Full page screenshot as base64 image data",
            "content": {
                "application/json": {
                    "example": {
                        "content_base64": "iVBORw0KGgoAAAANS...",
                        "content_type": "image/png",
                    }
                }
            },
        }
    },
)
async def get_pageshot(url: str):
    import base64

    target_url = f"{READER_BASE_URL}/{url}"
    headers = {"X-Respond-With": "pageshot"}
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                target_url, headers=headers, follow_redirects=True
            )
            response.raise_for_status()
            content_base64 = base64.b64encode(response.content).decode("utf-8")
            return JSONResponse(
                content={"content_base64": content_base64, "content_type": "image/png"}
            )
        except httpx.HTTPStatusError as e:
            raise HTTPException(status_code=e.response.status_code, detail=str(e))
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Internal server error: {str(e)}"
            )


@app.get(
    "/",
    summary="Retrieve URL content in LLM-friendly formats like markdown, text and others like screenshots.",
)
async def root():
    return {
        "message": "Reader Tools API is running.",
        "endpoints": {
            "markdown": "/markdown/{url}",
            "html": "/html/{url}",
            "text": "/text/{url}",
            "screenshot": "/screenshot/{url}",
            "pageshot": "/pageshot/{url}",
        },
    }


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"Internal server error: {str(exc)}"},
    )

