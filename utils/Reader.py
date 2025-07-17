import numpy as np
from pathlib import Path


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
        parts = header.split(",")
        # Skip header row if present
        if "timestamp" not in header:
            # First line already contains data
            i_t = 1
            i_o = 2
            i_l = 3

            if len(parts) >= 4:
                yield int(parts[i_t]), int(parts[i_o]), int(parts[i_l])
        else:
            i_t = parts.index("timestamp")
            i_o = parts.index("offset")
            i_l = parts.index("length")

        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 4:
                print("Warning")
                continue
            yield int(parts[i_t]), int(parts[i_o]), int(parts[i_l])


def decode_frame_normal_v2(payload, width, height):
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


def npz_generator(bin_name, root_path="data/"):
    if "bin" in bin_name:
        name = bin_name.split(".")[0]
        bin_path = f"{root_path}{bin_name}"
        info_path = f"{root_path}{name}_info.txt"

    else:
        name = bin_name
        bin_path = f"{root_path}{bin_name}.bin"
        info_path = f"{root_path}{name}_info.txt"


    BIN_FILE = Path(bin_path)
    INFO_FILE = Path(info_path)


    parts = name.split("_")
    WIDTH = int(parts[-3])
    HEIGHT = int(parts[-2])

    output_path = f"{root_path}events_{parts[-1]}.npz"
    OUT_FILE = Path(output_path)

    events_t, events_x, events_y, events_p = [], [], [], []

    with BIN_FILE.open("rb") as fh:
        for ts_us, offset, length in parse_info_file(INFO_FILE):
            fh.seek(offset)
            block = fh.read(length)

            # According to manual §7.9, payload is at the end of the block.
            expected_payload_len = (WIDTH * HEIGHT) // 4

            # print(f"帧结构长度:{length}\nsensor 数据长度：{expected_payload_len}")

            payload = block[128:128+expected_payload_len]
            x, y, p = decode_frame_normal_v2(payload, WIDTH, HEIGHT)

            # x, y, p = decode_frame_normal_v2(block, WIDTH, HEIGHT)
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
    npz_generator(bin_name="normal_v2_816_612_20250309170810117")
