import os
import argparse
import pickle
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager, rcParams

FONT_PROP = None


def linear_interpolate(coords, n_points: int):
    """Linearly interpolate a sequence of (lon, lat) pairs to n_points samples."""
    import numpy as _np
    coords = list(coords) if coords is not None else []
    if len(coords) == 0:
        return []
    arr = _np.asarray(coords, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return coords
    if len(arr) == n_points:
        return [tuple(row[:2]) for row in arr]
    if len(arr) > n_points:
        xs = _np.arange(len(arr))
        xs_new = _np.linspace(0, len(arr) - 1, n_points)
        lons = _np.interp(xs_new, xs, arr[:, 0])
        lats = _np.interp(xs_new, xs, arr[:, 1])
        return list(zip(lons.tolist(), lats.tolist()))
    xs = _np.arange(len(arr))
    xs_new = _np.linspace(0, len(arr) - 1, n_points)
    lons = _np.interp(xs_new, xs, arr[:, 0])
    lats = _np.interp(xs_new, xs, arr[:, 1])
    return list(zip(lons.tolist(), lats.tolist()))


def configure_chinese_font(font_path: str = None):
    global FONT_PROP
    found_fp = None
    if font_path:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
                name = font_manager.FontProperties(fname=font_path).get_name()
                fp = font_path
                found_fp = (fp, name)
            except Exception as e:
                print(f'Failed to load font from {font_path}: {e}')
                found_fp = None
        else:
            print(f'Font path does not exist: {font_path}')
            found_fp = None

    candidate_files = [
        r"C:\Windows\Fonts\msyh.ttc",
        r"C:\Windows\Fonts\msyh.ttf",
        r"C:\Windows\Fonts\simsunb.ttf",
        r"C:\Windows\Fonts\simsun.ttc",
        r"C:\Windows\Fonts\NotoSansCJK-Regular.ttc",
    ]
    for fp in candidate_files:
        if os.path.exists(fp):
            try:
                font_manager.fontManager.addfont(fp)
                name = font_manager.FontProperties(fname=fp).get_name()
                found_fp = (fp, name)
                break
            except Exception:
                continue

    if not found_fp:
        preferred_names = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK', 'STHeiti', 'Arial Unicode MS']
        found_name = None
        system_fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        for fp in system_fonts:
            try:
                name = font_manager.FontProperties(fname=fp).get_name()
            except Exception:
                continue
            for pn in preferred_names:
                if pn.lower() in name.lower():
                    found_name = name
                    found_fp = (fp, name)
                    break
            if found_fp:
                break

    if found_fp:
        fp, name = found_fp
        try:
            font_manager._rebuild()
        except Exception:
            pass
        rcParams['font.sans-serif'] = [name]
        rcParams['font.family'] = 'sans-serif'
        rcParams['axes.unicode_minus'] = False
        try:
            FONT_PROP = font_manager.FontProperties(fname=fp)
        except Exception:
            FONT_PROP = None
        print(f'Using font: {name} (from {fp})')
    else:
        rcParams['font.sans-serif'] = ['DejaVu Sans']
        rcParams['font.family'] = 'sans-serif'
        rcParams['axes.unicode_minus'] = False
        FONT_PROP = None
        print('Warning: No CJK-support font found. CJK characters may display as boxes.\n'
              'Please install Microsoft YaHei/SimHei or Noto Sans CJK, or pass a font file via --font-path.')


def list_system_fonts():
    fonts = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    out = []
    for fp in fonts:
        try:
            name = font_manager.FontProperties(fname=fp).get_name()
        except Exception:
            continue
        out.append((name, fp))
    out = sorted(out, key=lambda x: x[0])
    for name, fp in out:
        print(f'{name} -> {fp}')


def load_results(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def ensure_outdir(outdir: str):
    os.makedirs(outdir, exist_ok=True)


def get_coords_from_traj(traj) -> List[Tuple[float, float]]:
    if isinstance(traj, dict):
        if 'coords' in traj:
            return traj['coords']
        for key in ('trajectory', 'points', 'pts'):
            if key in traj:
                return traj[key]
    return traj


def plot_single_trajectory(results, outdir: str):
    first = results[0]
    coords = get_coords_from_traj(first)
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    plt.figure(figsize=(10, 6))
    plt.plot(lons, lats, 'b-', linewidth=1, label='Trajectory')
    plt.scatter(lons[::max(1, len(lons)//100)], lats[::max(1, len(lats)//100)],
                c='red', s=10, label='Sampled points')
    if FONT_PROP:
        plt.xlabel('Longitude', fontproperties=FONT_PROP)
        plt.ylabel('Latitude', fontproperties=FONT_PROP)
        plt.title('Single trajectory example', fontproperties=FONT_PROP)
        plt.legend(prop=FONT_PROP)
        ax = plt.gca()
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontproperties(FONT_PROP)
    else:
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Single trajectory example')
        plt.legend()
    plt.axis('equal')
    outpath = os.path.join(outdir, 'single_traj.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print('Saved', outpath)


def plot_overlay(results, outdir: str, max_trajs: int = 100):
    plt.figure(figsize=(10, 6))
    count = min(len(results), max_trajs)
    for i in range(count):
        coords = get_coords_from_traj(results[i])
        if not coords:
            continue
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        plt.plot(lons, lats, color='C0', alpha=0.25, linewidth=0.8)

    if FONT_PROP:
        plt.xlabel('Longitude', fontproperties=FONT_PROP)
        plt.ylabel('Latitude', fontproperties=FONT_PROP)
        plt.title(f'Trajectory overlay (first {count})', fontproperties=FONT_PROP)
        ax = plt.gca()
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontproperties(FONT_PROP)
    else:
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Trajectory overlay (first {count})')
    plt.axis('equal')
    outpath = os.path.join(outdir, 'overlay.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print('Saved', outpath)


def plot_density(results, outdir: str, gridsize: int = 200):
    all_lons = []
    all_lats = []
    for traj in results:
        coords = get_coords_from_traj(traj)
        if not coords:
            continue
        for c in coords:
            all_lons.append(c[0])
            all_lats.append(c[1])

    if len(all_lons) == 0:
        print('No points found for density plot.')
        return

    x = np.array(all_lons)
    y = np.array(all_lats)

    plt.figure(figsize=(8, 8))
    try:
        hb = plt.hexbin(x, y, gridsize=gridsize, cmap='inferno', mincnt=1)
        plt.colorbar(hb, label='Counts')
        if FONT_PROP:
            plt.xlabel('Longitude', fontproperties=FONT_PROP)
            plt.ylabel('Latitude', fontproperties=FONT_PROP)
            plt.title('Trajectory point density (Hexbin)', fontproperties=FONT_PROP)
            ax = plt.gca()
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_fontproperties(FONT_PROP)
        else:
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Trajectory point density (Hexbin)')
        plt.axis('equal')
        outpath = os.path.join(outdir, 'density_hexbin.png')
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        print('Saved', outpath)
    except Exception:
        plt.figure(figsize=(8, 8))
        plt.hist2d(x, y, bins=gridsize, cmap='inferno')
        plt.colorbar(label='Counts')
        if FONT_PROP:
            plt.xlabel('Longitude', fontproperties=FONT_PROP)
            plt.ylabel('Latitude', fontproperties=FONT_PROP)
            plt.title('Trajectory point density (hist2d)', fontproperties=FONT_PROP)
            ax = plt.gca()
            for tl in ax.get_xticklabels() + ax.get_yticklabels():
                tl.set_fontproperties(FONT_PROP)
        else:
            plt.xlabel('Longitude')
            plt.ylabel('Latitude')
            plt.title('Trajectory point density (hist2d)')
        plt.axis('equal')
        outpath = os.path.join(outdir, 'density_hist2d.png')
        plt.tight_layout()
        plt.savefig(outpath, dpi=200)
        plt.close()
        print('Saved', outpath)


def plot_scatter_sample(results, outdir: str, sample_rate: int = 50):
    xs = []
    ys = []
    for traj in results:
        coords = get_coords_from_traj(traj)
        if not coords:
            continue
        for c in coords[::sample_rate]:
            xs.append(c[0])
            ys.append(c[1])

    if not xs:
        print('No sampled points for scatter plot.')
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(xs, ys, s=6, alpha=0.6)
    if FONT_PROP:
        plt.xlabel('Longitude', fontproperties=FONT_PROP)
        plt.ylabel('Latitude', fontproperties=FONT_PROP)
        plt.title(f'Sampled scatter (every {sample_rate} points)', fontproperties=FONT_PROP)
        ax = plt.gca()
        for tl in ax.get_xticklabels() + ax.get_yticklabels():
            tl.set_fontproperties(FONT_PROP)
    else:
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title(f'Sampled scatter (every {sample_rate} points)')
    plt.axis('equal')
    outpath = os.path.join(outdir, 'scatter_sample.png')
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()
    print('Saved', outpath)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default='submission_val_input_8.pkl', help='结果 pickle 文件路径')
    parser.add_argument('--outdir', '-o', default='figures', help='输出图片目录')
    parser.add_argument('--max-trajs', type=int, default=200, help='用于覆盖图的最大轨迹数')
    parser.add_argument('--gridsize', type=int, default=200, help='热力图格子/分辨率')
    parser.add_argument('--font-path', type=str, default=None, help='指定字体文件路径（优先）')
    parser.add_argument('--interp-n', type=int, default=0, help='If >0, linearly interpolate each trajectory to this many points before plotting')
    parser.add_argument('--list-fonts', action='store_true', help='列出系统可用字体并退出')
    args = parser.parse_args()

    if args.list_fonts:
        list_system_fonts()
        return

    configure_chinese_font(args.font_path)

    results = load_results(args.input)
    if args.interp_n and args.interp_n > 0:
        interp_results = []
        for traj in results:
            coords = get_coords_from_traj(traj)
            interp_coords = linear_interpolate(coords, args.interp_n)
            interp_results.append(interp_coords)
        results = interp_results
    ensure_outdir(args.outdir)

    sns.set(style='whitegrid')

    if not results:
        print('No results loaded from', args.input)
        return

    plot_single_trajectory(results, args.outdir)
    plot_overlay(results, args.outdir, max_trajs=args.max_trajs)
    plot_density(results, args.outdir, gridsize=args.gridsize)
    plot_scatter_sample(results, args.outdir)


if __name__ == '__main__':
    main()
