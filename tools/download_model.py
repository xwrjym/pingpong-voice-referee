"""下载 Vosk 中文小模型（约 42MB，离线唤醒+固定口令足够用）。

    python tools/download_model.py

默认下到 <工程>/models/ 下，和 config.json 里的 model_dir 对齐。
国内网络慢的话可以加 --url 换成自己的镜像地址。
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import urllib.request
import zipfile

DEFAULT_URL = "https://alphacephei.com/vosk/models/vosk-model-small-cn-0.22.zip"
MODEL_NAME = "vosk-model-small-cn-0.22"


def _progress(block_count: int, block_size: int, total: int) -> None:
    if total <= 0:
        return
    done = min(block_count * block_size, total)
    percent = done * 100 / total
    bar = "#" * int(percent / 2)
    sys.stdout.write(f"\r  [{bar:<50}] {percent:5.1f}%  {done // 1048576}/{total // 1048576} MB")
    sys.stdout.flush()


def download(url: str, target: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    print(f"下载 {url}")
    urllib.request.urlretrieve(url, target, _progress)
    print()
    return target


def main() -> int:
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--models-dir", default=os.path.join(here, "models"))
    parser.add_argument("--force", action="store_true", help="已存在也重新下载")
    args = parser.parse_args()

    model_dir = os.path.join(args.models_dir, MODEL_NAME)
    if os.path.isdir(model_dir) and not args.force:
        print(f"模型已存在：{model_dir}")
        return 0

    zip_path = os.path.join(args.models_dir, os.path.basename(args.url))
    try:
        download(args.url, zip_path)
    except Exception as exc:
        print(f"下载失败：{exc}")
        print("可以手动下载后解压到 models/ 目录，保持目录名不变。")
        return 1

    print(f"解压到 {args.models_dir} ……")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.models_dir)
    os.remove(zip_path)

    if not os.path.isdir(model_dir):
        print(f"解压后没找到 {model_dir}，请检查压缩包结构")
        return 1
    size_mb = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, files in os.walk(model_dir) for name in files
    ) / 1048576
    print(f"完成：{model_dir}（{size_mb:.1f} MB）")
    vocab = os.path.join(model_dir, "graph", "words.txt")
    print(f"词典文件：{'有' if os.path.exists(vocab) else '无'}（无则会退化为全词表解码）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
