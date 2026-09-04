def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def parse_ts(value: str) -> float:
    parts = str(value).split(":")
    if len(parts) > 3:
        raise ValueError(f"bad timestamp: {value}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total
