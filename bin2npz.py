
#!/usr/bin/env python3
"""
convert_evs_bin_to_npz_fixed.py
================================
Standalone converter for APX EVB Gen2 ``normal_v2`` (normal_2bit) event streams.
All paths and sensor parameters are defined as constants below, so the script
can be run with a single command:

    python convert_evs_bin_to_npz_fixed.py

The output NPZ will contain four arrays:
    t : float64 – timestamps in seconds
    x : uint16  – x‐coordinates
    y : uint16  – y‐coordinates
    p : int8    – polarity (+1 for ON, −1 for OFF)
"""

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------------
# User‑defined paths and sensor parameters
# ---------------------------------------------------------------------------
BIN_FILE   = Path("data/normal_v2_816_612_20250309170420868.bin")
INFO_FILE  = Path("data/normal_v2_816_612_20250309170420868_info.txt")
WIDTH      = 816   # sensor width  in pixels
HEIGHT     = 612   # sensor height in pixels
OUT_FILE   = Path("data/events_20250309170420868.npz")
# ---------------------------------------------------------------------------


def parse_info_file(path):
    """
    Parse the *_info.txt table produced by APX SDK.

    Expected format:
        index,timestamp,offset,length
        0,12345678,0,124984
        1,12345728,124984,124984
        ...

    Yields tuples (timestamp_us, offset, length) in file order.
    """
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().strip()
        # Skip header row if present
        if "timestamp" not in header:
            # First line already contains data
            parts = header.split(",")
            if len(parts) >= 4:
                yield int(parts[1]), int(parts[2]), int(parts[3])
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                continue
            yield int(parts[1]), int(parts[2]), int(parts[3])


def decode_frame(payload, width, height):
    """
    Decode one normal_2bit frame.

    Each byte encodes four pixels (p0..p3), 2 bits per pixel:
        0 -> no event
        1 -> OFF event  (polarity -1)
        2 -> ON  event  (polarity +1)
        3 -> reserved   (ignored)

    Returns three 1‑D NumPy arrays: x, y, p
    """
    x_list, y_list, p_list = [], [], []
    total_pixels = width * height
    expected_len = total_pixels // 4
    payload = payload[:expected_len]  # ensure correct length

    for byte_idx, byte_val in enumerate(payload):
        if byte_val == 0:
            continue  # fast path: no events in this quad
        base_pixel = byte_idx * 4
        for sub in range(4):
            code = (byte_val >> (2 * sub)) & 0b11
            if code == 0 or code == 3:
                continue
            pixel_index = base_pixel + sub
            if pixel_index >= total_pixels:
                continue
            y = pixel_index // width
            x = pixel_index - y * width
            polarity = 1 if code == 2 else -1
            x_list.append(x)
            y_list.append(y)
            p_list.append(polarity)
    return (np.asarray(x_list, dtype=np.uint16),
            np.asarray(y_list, dtype=np.uint16),
            np.asarray(p_list, dtype=np.int8))


def main():
    events_t, events_x, events_y, events_p = [], [], [], []

    with BIN_FILE.open("rb") as fh:
        for ts_us, offset, length in parse_info_file(INFO_FILE):
            fh.seek(offset)
            block = fh.read(length)

            # According to manual §7.9, payload is at the end of the block.
            expected_payload_len = (WIDTH * HEIGHT) // 4
            payload = block[-expected_payload_len:]

            x, y, p = decode_frame(payload, WIDTH, HEIGHT)
            if x.size == 0:
                continue
            t = np.full_like(x, ts_us, dtype=np.float64) * 1e-6  # µs -> s

            events_t.append(t)
            events_x.append(x)
            events_y.append(y)
            events_p.append(p)

    # Concatenate all blocks
    if events_t:
        t = np.concatenate(events_t)
        x = np.concatenate(events_x)
        y = np.concatenate(events_y)
        p = np.concatenate(events_p)

        np.savez_compressed(OUT_FILE, t=t, x=x, y=y, p=p)
        print(f"Saved {t.size} events to '{OUT_FILE}'.")
    else:
        print("No events decoded; output file was not written.")


if __name__ == "__main__":
    main()
