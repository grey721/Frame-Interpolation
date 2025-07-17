import numpy as np
from pathlib import Path
from PIL import Image


def npz_to_png(npz_path: Path, png_path: Path, width: int, height: int,
               aim_frame: int = 0, polarity_map=None, normalize=False):
    """
    从 .npz 文件中读取 t,x,y,p 四个数组，按像素聚合事件，保存为灰度 PNG。

    参数：
    - npz_path: 输入的 .npz 文件路径
    - png_path: 输出的 .png 文件路径
    - width, height: 传感器分辨率
    - polarity_map: 可选 dict，将 p 值映射到灰度值，例如 {1:200, -1:100}
    - normalize: 如果为 True，则把计数归一化到 [0,255]，否则直接截断到 [0,255]
    """
    data = np.load(npz_path)
    t = data['t']
    x = data['x']
    y = data['y']
    p = data['p']

    last_time = t[0]
    current_frame = 0
    start: int = 0
    for i, now in enumerate(t):
        if now != last_time:
            current_frame += 1
            if current_frame == aim_frame:
                start = i
            elif current_frame > aim_frame:
                x = x[start:i]
                y = y[start:i]
                p = p[start:i]
                break
            last_time = now

    # 创建累积数组
    img = np.zeros((height, width), dtype=np.float32)

    if polarity_map is not None:
        # 用映射值填充
        for pol, grey in polarity_map.items():
            mask = (p == pol)
            coords = (y[mask], x[mask])
            img[coords] += grey
    else:
        # 默认：每个事件都 +1
        coords = (y, x)
        np.add.at(img, coords, 1)

    # 归一化或截断到 [0,255]
    if normalize:
        mx = img.max() if img.size > 0 else 1
        img = (img / mx) * 255
    img = np.clip(img, 0, 255).astype(np.uint8)

    # 保存
    Image.fromarray(img).save(png_path)
    print(f"[√] Saved image to {png_path} (shape={img.shape})")


def compare_pngs(png1_path, png2_path, diff_path="data/diff.png", save_diff=False):  # , amplify=False
    # 读取灰度图
    img1 = Image.open(png1_path).convert("L")
    img2 = Image.open(png2_path).convert("L")

    arr1 = np.array(img1, dtype=np.int16)
    arr2 = np.array(img2, dtype=np.int16)

    analyze_pixel_distribution(arr1, name="PNG1")
    analyze_pixel_distribution(arr2, name="PNG2")

    if arr1.shape != arr2.shape:
        raise ValueError(f"图像尺寸不一致: {arr1.shape} vs {arr2.shape}")

    # 差值：uint8 会自动截断负数，所以我们先用 int16
    diff = np.abs(arr1 - arr2).astype(np.uint8)

    if save_diff:
        # 如果希望对比更明显，可以放大差值（视觉增强用，不参与统计）
        # vis_diff = diff.copy()
        # if amplify:
        #     vis_diff = np.clip(diff * 10, 0, 255).astype(np.uint8)

        # 保存差值图
        Image.fromarray(diff).save(diff_path)

    # 统计差异
    print(f"\n[✓] Saved diff image to '{diff_path}'")

    nonzero_diff = diff[diff != 0]
    if nonzero_diff.size > 0:
        print(f"平均差值（mean diff）: {nonzero_diff.mean():.2f}")
        print(f"最大差值（max diff） : {nonzero_diff.max()}")
        print(f"最小非零差值: {nonzero_diff.min()}")
    else:
        print("所有像素完全一致")


def analyze_pixel_distribution(img_array, name="图像"):
    """
    统计图像中所有像素值的分布及占比
    """
    values, counts = np.unique(img_array, return_counts=True)
    total = img_array.size

    print(f"\n{name} 像素值组成分析（共 {total} 像素）:")
    for val, count in zip(values, counts):
        ratio = count / total * 100
        print(f"  值 {val:3d}: {count:8d} 像素，占比 {ratio:6.2f}%")


if __name__ == "__main__":
    # 修改如下路径和参数
    get_frame = 0
    npz_file = Path("data/events_20250309170810117.npz")
    png_file = Path(f"data/events_20250309170810117_{get_frame}.png")
    WIDTH, HEIGHT = 816, 612  # 注意和你 parse 时的顺序一致！
    # 如果想区分 ON/ OFF 事件，传入 polarity_map：
    polarity_map = {1: 200, -1: 100}
    # polarity_map = None
    npz_to_png(npz_file, png_file, WIDTH, HEIGHT,
               polarity_map=polarity_map, aim_frame=get_frame)

    png2 = Path("data/816_612_8_0000000000.png")  # 假设是官方软件导出的
    diff_png = "data/diff.png"

    compare_pngs(png_file, png2, diff_png, save_diff=True)
