"""Agent tools: weather lookup and web search."""

import re
import httpx
from duckduckgo_search import DDGS
from langchain_core.tools import tool

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (Open-Meteo)
WEATHER_DESCRIPTIONS: dict[int, str] = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _weather_condition(code: int) -> str:
    """Return a human-readable weather condition for a WMO code."""
    return WEATHER_DESCRIPTIONS.get(code, f"Weather code {code}")


def _parse_coordinates(location: str) -> tuple[float, float, str] | None:
    """Parse 'latitude,longitude' from a location string."""
    match = re.match(
        r"^\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*$",
        location.strip(),
    )
    if not match:
        return None
    latitude = float(match.group(1))
    longitude = float(match.group(2))
    label = f"{latitude}, {longitude}"
    return latitude, longitude, label


def _geocode_location(location: str) -> tuple[float, float, str]:
    """Resolve a place name to coordinates using Open-Meteo geocoding."""
    try:
        response = httpx.get(
            GEOCODING_URL,
            params={"name": location.strip(), "count": 1, "language": "en", "format": "json"},
            timeout=8.0, # Optimized timeout
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("results") or []
    except Exception as e:
        raise ValueError(f"Geocoding service unavailable or failed: {str(e)}")

    if not results:
        raise ValueError(f"Could not find location coordinates for: {location}")

    place = results[0]
    latitude = float(place["latitude"])
    longitude = float(place["longitude"])
    name = place.get("name", location)
    country = place.get("country")
    label = f"{name}, {country}" if country else name
    return latitude, longitude, label


def _fetch_weather(location: str) -> str:
    """Fetch live weather data from Open-Meteo."""
    coords = _parse_coordinates(location)
    if coords is None:
        coords = _geocode_location(location)

    latitude, longitude, label = coords

    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            "timezone": "auto",
        },
        timeout=8.0,
    )
    response.raise_for_status()
    current = response.json().get("current") or {}

    temperature = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")
    wind_speed = current.get("wind_speed_10m")
    weather_code = current.get("weather_code", -1)

    if temperature is None:
        raise ValueError(f"Weather data unavailable for {label}.")

    condition = _weather_condition(int(weather_code))
    return (
        f"Weather for {label}:\n"
        f"- Temperature: {temperature}°C\n"
        f"- Condition: {condition}\n"
        f"- Humidity: {humidity}%\n"
        f"- Wind speed: {wind_speed} km/h"
    )


def _search_web(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo backend safely."""
    try:
        # DDGS initialized with a safe execution timeout
        with DDGS(timeout=10) as ddgs:
            raw_results = ddgs.text(str(query), max_results=max_results)
            results = list(raw_results) if raw_results else []
    except Exception as e:
        return f"DuckDuckGo search pipeline execution error: {str(e)}"

    if not results:
        return f"No web results found for the query: {query}"

    lines = [f"Web search results for: {query}"]
    for index, result in enumerate(results, start=1):
        title = result.get("title", "Untitled Source")
        snippet = result.get("body") or result.get("text") or result.get("snippet") or ""
        snippet = snippet.strip()
        url = result.get("href") or result.get("link") or ""
        
        lines.append(f"{index}. {title}")
        if snippet:
            lines.append(f"   {snippet}")
        if url:
            lines.append(f"   Source: {url}")

    return "\n".join(lines)


@tool
def get_weather(location: str) -> str:
    """Get current live weather for a city or coordinates (latitude,longitude).

    Use this tool ONLY when the user explicitly asks for current weather conditions, temperatures, or local forecasts.
    """
    try:
        return _fetch_weather(str(location))
    except httpx.HTTPError as exc:
        return f"Weather lookup failed due to a network error: {exc}"
    except ValueError as exc:
        return f"Weather lookup failed: {exc}"


@tool
def web_search(query: str) -> str:
    """Search the internet for real-time information, news, live sports scores, and modern facts.

    Use this tool ONLY when the question requires live internet facts or information that cannot be answered with static knowledge.
    """
    try:
        return _search_web(str(query))
    except Exception as exc:
        return f"Web search tool execution failed completely: {exc}"


AGENT_TOOLS = [get_weather, web_search]