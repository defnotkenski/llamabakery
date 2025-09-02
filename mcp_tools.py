import csv
from pathlib import Path


def get_weather(loc: str) -> str:
    """
    Get the current temperature by city name.

    Use only when the user explicitly asks about weather/temperature/forecast.
    Otherwise, answer normally without mentioning or calling tools.

    Args:
      loc (str): The name of the city without the state. (e.g., "Carmel")

    Returns:
      str: The current temperature along with city name.
    """

    return f"{loc}: 67F"


def remember_event(name: str, time: str) -> str:
    """
    Remember an event.

    Use only when the user explicitly mentions an event.
    Otherwise, answer normally without mentioning or calling tools.

    Args:
      name (str): The name of the event. (e.g., "football practice")
      time (str): Time of the event ending. (e.g., "4pm"

    Returns:
      str: Whether or not the tool was successful.
    """
    p = Path.cwd().joinpath("mock_db.csv")

    fields = ["name", "time"]
    row = {"name": name, "time": time}

    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(row)

    return "success"
