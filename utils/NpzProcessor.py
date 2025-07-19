import numpy as np
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm


class EventFlowProcessor:
    def __init__(self, npz_name, height, width, root_path="data/"):
        if ".npz" in npz_name:
            npz_name = npz_name.split(".")[0]
        npz_path = Path(f"{root_path}{npz_name}.npz")

        self.npz_name = npz_name
        with np.load(npz_path) as data:
            self.data = data
        self.root_path = root_path
        os.makedirs(root_path, exist_ok=True)
        self.h = height
        self.w = width

    def make_event_frame_npzs(self, fps,
                              delta: int = None,
                              save_empty: bool = True, save_png: bool = False,
                              polarity_map: dict = None
                              ) -> None:
        """
        :param fps: 期望帧率
        :param delta: 取事件帧时的窗口时间
        :param save_empty:是否保存空事件帧
        :param save_png:是否输出png
        :param polarity_map:生成灰度图png时的像素值映射
        :return:
        """
        t = self.data['t']
        x = self.data['x'].astype(np.int64)
        y = self.data['y'].astype(np.int64)
        p = self.data['p']

        total_event = np.unique(t).size
        if fps > total_event:
            raise ValueError(f"无法提供该帧率的输出，当前总的事件数量为：{total_event}")

        root = f"{self.root_path}{self.npz_name}_event_frame"
        os.makedirs(root, exist_ok=True)

        npz_path = f"{root}/npz"
        os.makedirs(npz_path, exist_ok=True)

        png_path = f"{root}/png"
        os.makedirs(png_path, exist_ok=True)

        t0, tn = t.min(), t.max()
        dt = 1e6 / fps  # µs per frame
        if delta:
            delta = max(dt, delta)

        num_frames = int(np.ceil((t.max() - t0) / dt))
        for i in range(num_frames):
            t_end = t0 + (i + 1) * dt
            t_start = t_end - (delta if delta else dt)

            if t_end == tn:
                mask = (t >= t_start) & (t <= t_end)
            else:
                mask = (t >= t_start) & (t < t_end)

            if not np.any(mask) and not save_empty:
                continue

            sub = {
                't': t[mask],
                'x': x[mask],
                'y': y[mask],
                'p': p[mask],
            }
            np.savez_compressed(f"{npz_path}/{i:04d}.npz", **sub)

            if save_png:
                # 创建累积数组
                img = np.zeros((self.h, self.w), dtype=np.float32)

                if polarity_map is not None:
                    # 用映射值填充
                    for pol, grey in polarity_map.items():
                        mask = (sub['p'] == pol)
                        # if np.any(mask):
                        coords = (sub['y'][mask], sub['x'][mask])
                        img[coords] += grey
                else:
                    # 默认：每个事件都 +1
                    coords = (sub['y'], sub['x'])
                    np.add.at(img, coords, 255)

                img = np.clip(img, 0, 255).astype(np.uint8)

                # 保存
                Image.fromarray(img).save(f"{png_path}/{i:04d}.png")


def merge_npz_event_dir(npz_dir, out_path, sort=False):
    """
    将一个目录下所有 .npz 文件中的事件 (t, x, y, p) 合并为一个大 npz 文件保存。

    参数：
        npz_dir (str or Path): 目录路径，包含多个 npz 文件
        out_path (str or Path): 合并后输出文件路径
        sort (bool): 是否按时间戳排序
    """
    npz_dir = Path(npz_dir)
    out_path = Path(out_path)
    files = sorted(npz_dir.glob("*.npz"))

    all_t, all_x, all_y, all_p = [], [], [], []

    for file in tqdm(files, desc="合并 npz 文件"):
        with np.load(file) as data:
            all_t.append(data['t'])
            all_x.append(data['x'])
            all_y.append(data['y'])
            all_p.append(data['p'])

    # 合并所有 npz 中的数据
    t = np.concatenate(all_t)
    x = np.concatenate(all_x)  # .astype(np.int64)
    y = np.concatenate(all_y)  # .astype(np.int64)
    p = np.concatenate(all_p)

    if sort:
        sort_idx = np.argsort(t)
        t = t[sort_idx]
        x = x[sort_idx]
        y = y[sort_idx]
        p = p[sort_idx]

    # 保存为压缩 npz
    print('压缩保存中……')
    np.savez_compressed(out_path, t=t, x=x, y=y, p=p)
    print(f"[√] 合并完成，{np.unique(t).size}个瞬间中共有 {len(t)} 个事件，已保存到：{out_path}")


if __name__ == "__main__":
    root = 'data/'
    npz_dir_name = 'RGB-EVS'
    for i in range(6):
        video_root = f'Video{i:03d}/'
        npz_root = f'{root}{video_root}{npz_dir_name}/'
        out_file = f'{root}{video_root}{npz_dir_name}/all.npz'
        merge_npz_event_dir(npz_dir=npz_root, out_path=out_file)
    # video_root = f'Video006/'
    # npz_root = f'{root}{video_root}{npz_dir_name}/'
    # out_file = f'{root}{video_root}{npz_dir_name}/all.npz'
    # merge_npz_event_dir(npz_dir=npz_root, out_path=out_file)
