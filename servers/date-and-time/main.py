"""
Date and Time Tools API.

Provides an API for LLMs to retrieve and calculate date and time information, including:
- Getting the current UTC time
- Getting the server's local time
- Getting the current time in any IANA timezone
- Calculating elapsed time between two timestamps in various units (seconds, minutes, hours, days)

Can return structured, human-readable results and handle timezone and DST details. Suitable for LLM function calling.

Author: bgeneto
Date: 2025-04-27
Version: 1.1.1
Last Modified: 2025-04-27
"""

from fastapi import FastAPI, HTTPException, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from datetime import datetime, timezone
from typing import Literal, Optional, Dict, Any
import logging
import traceback

# Supported languages for localization
SUPPORTED_LANGUAGES = {"en", "es", "pt-BR"}

# Configure logging for debugging and error tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("date-time-tools")

app = FastAPI(
    title="Date & Time Tools",
    version="1.1.1",
    description="Provides an API for LLMs to retrieve and calculate date and time information, including current time in UTC, local, or any IANA timezone, and elapsed time calculations.",
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


class ElapsedTimeInput(BaseModel):
    """
    Input model for elapsed time calculation.
    - start: ISO 8601 timestamp (timezone-aware or assumed UTC)
    - end: ISO 8601 timestamp (timezone-aware or assumed UTC)
    - units: Unit for elapsed time (seconds, minutes, hours, days; defaults to seconds)
    - language: Language code for human-readable output (e.g., 'en', 'es')
    """

    start: datetime = Field(
        ...,
        description="ISO 8601 timestamp, timezone-aware or assumed UTC. Accepts Unix timestamp as integer.",
        examples=["2024-06-01T12:00:00Z", 1717233600],
    )
    end: datetime = Field(
        ...,
        description="ISO 8601 timestamp, timezone-aware or assumed UTC. Accepts Unix timestamp as integer.",
        examples=["2024-06-01T13:00:00Z", 1717237200],
    )
    units: Optional[Literal["seconds", "minutes", "hours", "days"]] = Field(
        None,
        description="Unit for elapsed time (defaults to seconds)",
        examples=["minutes"],
    )
    language: Optional[str] = Field(
        "en",
        description="Language code for human-readable output (e.g., 'en', 'es', 'pt'). Defaults to English.",
        examples=["en", "es", "pt"],
    )


# Response model for time queries
class TimeResult(BaseModel):
    """
    Structured response for time queries.
    - timezone: IANA timezone name
    - datetime: ISO 8601 timestamp
    - unix_timestamp: Unix timestamp (seconds since epoch)
    - is_dst: Whether Daylight Saving Time is active
    - human_readable: Human-friendly string representation
    """

    timezone: str
    datetime: str
    unix_timestamp: int = Field(..., description="Unix timestamp (seconds since epoch)")
    is_dst: bool
    human_readable: str


# Helper for localization
HUMAN_TEMPLATES = {
    "en": "{weekday}, {day} {month} {year}, {hour}:{minute}:{second} {tz}",
    "es": "{weekday}, {day} de {month} de {year}, {hour}:{minute}:{second} {tz}",
    "pt-BR": "{weekday}, {day} de {month} de {year}, {hour}:{minute}:{second} {tz}",
}
WEEKDAYS = {
    "en": [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ],
    "es": ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"],
    "pt-BR": [
        "Segunda-feira",
        "Terça-feira",
        "Quarta-feira",
        "Quinta-feira",
        "Sexta-feira",
        "Sábado",
        "Domingo",
    ],
}
MONTHS = {
    "en": [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ],
    "es": [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ],
    "pt-BR": [
        "janeiro",
        "fevereiro",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    ],
}


def humanize(dt: datetime, tz: str, lang: str) -> str:
    # Ensure language code is supported, fallback to 'en'
    lang = lang if lang in SUPPORTED_LANGUAGES else "en"
    template = HUMAN_TEMPLATES[lang]
    # Python's weekday(): Monday=0, Sunday=6; month: 1-12
    weekday = WEEKDAYS[lang][dt.weekday()]
    month = MONTHS[lang][dt.month - 1]
    # Special case for pt-BR: lowercase weekday/month except first letter
    if lang == "pt-BR":
        weekday = weekday.capitalize()
        month = month.lower()
    return template.format(
        weekday=weekday,
        day=dt.day,
        month=month,
        year=dt.year,
        hour=str(dt.hour).zfill(2),
        minute=str(dt.minute).zfill(2),
        second=str(dt.second).zfill(2),
        tz=tz,
    )


@app.get(
    "/get_current_utc_time",
    summary="Current UTC time",
    operation_id="get_current_utc_time",
    response_model=TimeResult,
    responses={
        200: {
            "description": "Current UTC time in multiple formats.",
            "content": {
                "application/json": {
                    "example": {
                        "timezone": "UTC",
                        "datetime": "2024-06-01T12:00:00+00:00",
                        "unix_timestamp": 1717233600,
                        "is_dst": False,
                        "human_readable": "Saturday, 01 June 2024, 12:00:00 UTC",
                    }
                }
            },
        }
    },
)
def get_current_utc(
    language: str = Query(
        "en",
        description="Language code for human-readable output (e.g., 'en', 'es', 'pt').",
    )
):
    """
    Returns the current time in UTC in ISO format, Unix timestamp, with DST info and a human-readable string (localized).
    """
    now = datetime.now(timezone.utc)
    human = humanize(now, "UTC", language)
    return TimeResult(
        timezone="UTC",
        datetime=now.isoformat(timespec="seconds"),
        unix_timestamp=int(now.timestamp()),
        is_dst=False,
        human_readable=human,
    )


@app.get(
    "/get_current_local_time",
    summary="Current Local Time",
    operation_id="get_current_local_time",
    response_model=TimeResult,
    responses={
        200: {
            "description": "Current local time in multiple formats.",
            "content": {
                "application/json": {
                    "example": {
                        "timezone": "Europe/Berlin",
                        "datetime": "2024-06-01T14:00:00+02:00",
                        "unix_timestamp": 1717240800,
                        "is_dst": True,
                        "human_readable": "Sábado, 01 de junio de 2024, 14:00:00 Europe/Berlin",
                    }
                }
            },
        }
    },
)
def get_current_local(
    language: str = Query(
        "en", description="Language code for human-readable output (e.g., 'en', 'es')."
    )
):
    """
    Returns the current local time (server's local time) in ISO format, Unix timestamp, with DST info and a human-readable string (localized).
    Note: This is the server's local time, not necessarily the user's.
    """
    now = datetime.now().astimezone()
    is_dst = now.dst() is not None and now.dst().total_seconds() > 0
    tzname = str(now.tzinfo)
    human = humanize(now, tzname, language)
    return TimeResult(
        timezone=tzname,
        datetime=now.isoformat(timespec="seconds"),
        unix_timestamp=int(now.timestamp()),
        is_dst=is_dst,
        human_readable=human,
    )


@app.post(
    "/elapsed_time",
    summary="Time elapsed between timestamps",
    operation_id="elapsed_time",
    response_model=Dict[str, Any],
    responses={
        200: {
            "description": "Elapsed time between two timestamps.",
            "content": {
                "application/json": {"example": {"elapsed": 3600, "unit": "seconds"}}
            },
        },
        400: {
            "description": "Invalid input.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "End must be after start. Use ISO 8601 or Unix timestamp."
                    }
                }
            },
        },
    },
)
def elapsed_time(data: ElapsedTimeInput):
    """
    Calculates the elapsed time between two timestamps in the specified unit (seconds, minutes, hours, days).
    Accepts ISO 8601 or Unix timestamps. Treats naive datetimes as UTC.
    """

    # Accept Unix timestamps as int
    def parse_dt(val):
        if isinstance(val, int):
            return datetime.fromtimestamp(val, tz=timezone.utc)
        return val if val.tzinfo else val.replace(tzinfo=timezone.utc)

    start = parse_dt(data.start)
    end = parse_dt(data.end)
    if end < start:
        raise HTTPException(
            status_code=400,
            detail="End must be after start. Use ISO 8601 or Unix timestamp.",
        )
    delta = end - start
    total = delta.total_seconds()
    unit = data.units or "seconds"
    mapping = {
        "seconds": total,
        "minutes": total / 60,
        "hours": total / 3600,
        "days": total / 86400,
    }
    return {"elapsed": mapping[unit], "unit": unit}


@app.get(
    "/get_current_time_by_timezone",
    summary="Current time in a specific IANA timezone",
    response_model=TimeResult,
    operation_id="get_current_time_by_timezone",
    responses={
        200: {
            "description": "Current time in a specific IANA timezone.",
            "content": {
                "application/json": {
                    "example": {
                        "timezone": "America/New_York",
                        "datetime": "2024-06-01T08:00:00-04:00",
                        "unix_timestamp": 1717228800,
                        "is_dst": True,
                        "human_readable": "Saturday, 01 June 2024, 08:00:00 America/New_York",
                    }
                }
            },
        },
        400: {
            "description": "Invalid timezone.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Invalid or unknown IANA timezone name: 'Mars/Phobos'. Try a valid timezone like 'America/New_York'."
                    }
                }
            },
        },
    },
)
def get_current_time_by_timezone(
    timezone_name: str = Query(
        ...,
        description="IANA timezone name (e.g., 'Asia/Tokyo', 'America/New_York', 'Europe/London')",
        examples=["America/Los_Angeles", "UTC"],
    ),
    language: str = Query(
        "en", description="Language code for human-readable output (e.g., 'en', 'es')."
    ),
):
    """
    Returns the current time in the specified IANA timezone, including DST information, Unix timestamp, and a human-readable string (localized).
    """
    try:
        tz = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid or unknown IANA timezone name: '{timezone_name}'. Try a valid timezone like 'America/New_York'.",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing timezone '{timezone_name}': {str(e)}",
        )
    current_time = datetime.now(tz)
    is_dst_active = (
        current_time.dst() is not None and current_time.dst().total_seconds() > 0
    )
    human = humanize(current_time, timezone_name, language)
    return TimeResult(
        timezone=timezone_name,
        datetime=current_time.isoformat(timespec="seconds"),
        unix_timestamp=int(current_time.timestamp()),
        is_dst=is_dst_active,
        human_readable=human,
    )


# Global exception handler for logging and returning a generic error message
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}\n{traceback.format_exc()}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": f"Internal server error: {str(exc)}"},
    )

