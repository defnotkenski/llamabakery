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
    Remember an event for later reference or follow-up.
    Call this tool whenever the user mentions any upcoming or planned event, activity, or appointment—even if it's incidental, not the main topic, or just venting (e.g., "I have football practice at 4pm" or "Doctor's appointment tomorrow morning"). This helps build realism in conversations, like an AI companion remembering details to ask about later.

    Do not call for past events, vague references without details, or if no time/context is implied. If the time is unclear, use a placeholder like "later today" or "unspecified time".

    Args:
      name (str): A descriptive name for the event (e.g., "football practice", "doctor's appointment")
      time (str): The time or timeframe of the event (e.g., "4pm", "tomorrow morning")

    Returns:
      str: Whether or not the tool was successful (e.g., "success").
    """
    p = Path.cwd().joinpath("mock_db.csv")

    fields = ["name", "time"]
    row = {"name": name, "time": time}

    with p.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writerow(row)

    return "success"
