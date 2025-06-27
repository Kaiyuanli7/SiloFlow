import argparse
import json
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("schedule_config.json")

parser = argparse.ArgumentParser(
    description="Update weekly training schedule for the GranaryPredict service.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "--day", "--day-of-week", dest="day_of_week", default="sun",
    help="Cron day_of_week field (e.g. sun, mon-fri, *).",
)
parser.add_argument("--hour", type=int, default=2, help="Hour (0-23)")
parser.add_argument("--minute", type=int, default=0, help="Minute (0-59)")

args = parser.parse_args()

cfg = {"day_of_week": args.day_of_week, "hour": args.hour, "minute": args.minute}
CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
print(f"Schedule updated → {CONFIG_PATH} : {cfg}") 