import os
import logging
import traceback
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
import httpx

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("wolfram-llm")

# FastAPI app
app = FastAPI(
    title="Wolfram LLM Tools",
    description="Provides math- and science-related answers using Wolfram|Alpha LLM API",
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


class QueryInput(BaseModel):
    query: str = Field(
        ...,
        description="The query text. WolframAlpha understands natural language queries about entities in science and also math expressions.",
    )


# =========================
#   WolframLLMService
# =========================


class WolframLLMService:
    """
    Service for querying the WolframAlpha LLM API and returning raw text responses.
    """

    def __init__(self, app_id: str):
        self.app_id = app_id
        self.base_url = "https://api.wolframalpha.com/v1/llm-api"

    async def query(self, input_query: str) -> str:
        """
        Query the WolframAlpha LLM API and return the raw text response.
        Args:
            input_query (str): The user's query string.
        Returns:
            str: The raw text response from WolframAlpha LLM API.
        Raises:
            Exception: If the API call fails or returns an error.
        """
        params = {"appid": self.app_id, "input": input_query}
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.get(self.base_url, params=params)
                response.raise_for_status()
                return response.text
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error from WolframAlpha LLM API: {e.response.status_code} - {e.response.text}"
            )
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"WolframAlpha API error: {e.response.text}",
            )
        except Exception as e:
            logger.error(f"Unexpected error from WolframAlpha LLM API: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal error contacting WolframAlpha LLM API.",
            )


wolfram_llm_service = WolframLLMService(app_id=os.environ.get("WOLFRAM_LLM_APP_ID", ""))


@app.post(
    "/ask_wolfram_alpha",
    summary="Ask WolframAlpha queries about entities in math, chemistry, physics, astronomy, and science in general.",
    operation_id="ask_wolfram_alpha",
)
async def ask_wolfram(input: QueryInput):
    """
    Endpoint to query WolframAlpha LLM API and return the raw text response.
    """
    if not input.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    result = await wolfram_llm_service.query(input.query)
    return {"success": True, "result": result}


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"Internal server error: {str(exc)}"},
    )
