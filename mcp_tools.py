def weather_penis(loc: str) -> str:
    """
    Get the current temperature by city name.

    Use only when the user explicitly asks about weather/temperature/forecast
    for a location. Otherwise, do not use this tool and answer normally.

    Args:
      loc (str): The name of the city without the state. (e.g., "Carmel")

    Returns:
      str: The current temperature along with city name.
    """

    return f"{loc}: 67F"
