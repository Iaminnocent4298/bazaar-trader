from datetime import datetime, timedelta, timezone

from fetch_history import fetch_history
from pool import POOL

end = datetime.now(timezone.utc)
start = end - timedelta(hours=6)

for tag in POOL:
    try:
        rows = fetch_history(tag, start, end)
        print(f"{tag}: {len(rows)} rows")
    except Exception as e:
        print(f"{tag}: FAILED ({e})")