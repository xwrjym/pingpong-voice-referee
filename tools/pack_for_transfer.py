"""把工程打包成一个可以直接拷到另一台电脑的压缩包。

    python tools/pack_for_transfer.py                     # 不带模型（目标机自己下）
    python tools/pack_for_transfer.py --with-model        # 连 Vosk 模型一起带上（约 25MB 压缩后）
    python tools/pack_for_transfer.py --with-testdata     # 连合成测试集也带上

包里会附一份 BUNDLE_README.md，目标电脑上解压后跑 setup.ps1 即可。
不含 .venv（虚拟环境跟机器绑定，目标机重建）、runtime/、realdata/。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INCLUDE_DIRS = ["tt_voice", "tt_referee", "tools", "tests"]
INCLUDE_FILES = ["config.json", "requirements.txt", "README.md", "setup.ps1", ".gitignore",
                 "HANDOFF_PROMPT.md", "HANDOFF_PROMPT_ROUND2.md",
                 "语音纠错模块复现报告.md"]
EXCLUDE_PARTS = {".venv", "__pycache__", "runtime", "realdata", ".git", "node_modules"}

# 报告里引用的两套真人录音（v0.2 对照基线、v0.3 最终验收）
REALDATA_DIRS = ("realdata", "realdata4")

BUNDLE_README = """# 乒乓球裁判 · 离线语音唤醒与固定口令识别（搬运包）

## 一、在这台新电脑上装好

1. 把整个文件夹解压到任意路径（**路径尽量别带空格和中文**，避免奇怪的编码问题）。
2. 需要 Python 3.10 或更高版本（3.11 最稳），安装时勾选 "Add python.exe to PATH"。
3. 右键 `setup.ps1` → 使用 PowerShell 运行；或者在该目录打开 PowerShell 执行：

   ```powershell
   powershell -ExecutionPolicy Bypass -File setup.ps1
   ```

   脚本会自动：建虚拟环境 → 装依赖（阿里云镜像）→ 检查/下载识别模型 → 列麦克风 → 自检 → 跑一遍文本演示。

## 二、没有模型怎么办

* 包大小 > 40MB 说明模型已经带在包里（`models/vosk-model-small-cn-0.22`），跳过下载；
* 包很小的话，脚本会尝试联网下载（约 42MB，国内建议挂代理）；
* 没有网络也可以直接从原来的电脑把 `models\\` 整个目录拷过来；
* 或者改用零模型后端：先 `tools\\enroll.py --mic --repeats 3` 注册模板，
  再给命令加 `--engine dtw`，或把 `config.json` 里的 `engine` 改成 `"dtw"`。

## 三、怎么测

```powershell
# 1) 看麦克风编号
.venv\\Scripts\\python.exe -m tt_voice.cli devices

# 2) 实时跑（打不开默认麦克风就加 --device <编号>）
.venv\\Scripts\\python.exe -m tt_voice.cli mic --say --verbose
#    说"小裁判"→ 提示音/播报"在" → 6 秒内说"我加一分""刚才不算""我几比几"

# 3) 真人验收：按提示录 60 多条（唤醒/口令/连说/干扰），然后出指标
.venv\\Scripts\\python.exe tools\\record_dataset.py --out realdata --device <编号>
.venv\\Scripts\\python.exe -m tt_voice.cli eval realdata

# 4) 结果与日志（带回来复盘）
#    realdata\\eval_report_*.csv  逐条明细
#    runtime\\events.jsonl        每次识别/执行记录
```

## 四、注意事项

* 播报期间识别会自动静音（半双工），避免机器人听到自己说话；支持 AEC 的阵列麦可以关掉这个限制。
* 现场噪声大就调 `config.json` 里的 `vad_margin_db`（调大更保守）和 `match_threshold`。
* 唤醒词可以说「小裁判」；想加「小乒乓」等第二个唤醒词，直接写进 `wake_words` 数组。
* 换人使用不需要重注册（默认 Vosk 后端与说话人无关）。

## 五、随包真人录音（可选）

如果包内出现 `realdata/` 或 `realdata4/`，它们是两份本人录音的验收集：

* `realdata/`：v0.2，61 条，蓝牙耳机麦克风，对应报告里 89.8% 那一列；
* `realdata4/`：v0.3，73 条，手机经 UU 远控作麦克风，对应报告里 100% 那一列；
* 两个目录都带 `labels.json`，直接执行 `python -m tt_voice.cli eval realdata4` 即可复现报告数字。

这些是个人语音数据，仅用于本项目验收与复现；对外公开分发或交给无关人员前请先删除这两个目录。
"""


def iter_files(with_model: bool, with_testdata: bool, with_realdata: bool = False):
    for name in INCLUDE_FILES:
        path = os.path.join(PROJECT_ROOT, name)
        if os.path.exists(path):
            yield path
    for directory in INCLUDE_DIRS:
        root = os.path.join(PROJECT_ROOT, directory)
        for current, dirs, files in os.walk(root):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_PARTS]
            for name in files:
                if name.endswith(".pyc"):
                    continue
                yield os.path.join(current, name)
    # 模板库很小，默认带上（DTW 后端的演示用）
    dtw_dir = os.path.join(PROJECT_ROOT, "models", "dtw_cn")
    if os.path.isdir(dtw_dir):
        for name in os.listdir(dtw_dir):
            yield os.path.join(dtw_dir, name)
    if with_model:
        model = os.path.join(PROJECT_ROOT, "models", "vosk-model-small-cn-0.22")
        for current, dirs, files in os.walk(model):
            for name in files:
                yield os.path.join(current, name)
    if with_testdata:
        data = os.path.join(PROJECT_ROOT, "testdata")
        for current, dirs, files in os.walk(data):
            dirs[:] = [d for d in dirs if d not in EXCLUDE_PARTS]
            for name in files:
                yield os.path.join(current, name)
    if with_realdata:
        # 真人验收集：realdata/（v0.2，61 条）、realdata4/（v0.3，73 条）等
        for entry in REALDATA_DIRS:
            folder = os.path.join(PROJECT_ROOT, entry)
            if not os.path.isdir(folder):
                continue
            if not os.path.exists(os.path.join(folder, "labels.json")):
                continue
            for current, dirs, files in os.walk(folder):
                dirs[:] = [d for d in dirs if d not in EXCLUDE_PARTS]
                for name in files:
                    if name.endswith((".wav", ".json", ".csv", ".txt")):
                        yield os.path.join(current, name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(os.path.dirname(PROJECT_ROOT),
                                                      "tt_voice_transfer.zip"))
    parser.add_argument("--with-model", action="store_true", help="把 Vosk 模型一起打包")
    parser.add_argument("--with-testdata", action="store_true", help="把合成测试集一起打包")
    parser.add_argument("--with-realdata", action="store_true",
                        help="把真人验收集（realdata*，含录音与标签）一起打包")
    args = parser.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    files = sorted(set(iter_files(args.with_model, args.with_testdata, args.with_realdata)))
    if not files:
        print("没有找到可打包的文件")
        return 1

    with zipfile.ZipFile(out_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in files:
            arcname = os.path.relpath(path, PROJECT_ROOT).replace("\\", "/")
            zf.write(path, arcname)
        zf.writestr("BUNDLE_README.md", BUNDLE_README)

    size_mb = os.path.getsize(out_path) / 1048576
    digest = hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:16]
    print(f"已打包 {len(files)} 个文件 → {out_path}")
    print(f"大小 {size_mb:.1f} MB，sha256(前16位) {digest}")
    if not args.with_model:
        print("提示：没有带 Vosk 模型，目标电脑需要联网下载或从本机拷 models\\ 目录")
    print("\n在目标电脑上：解压 → powershell -ExecutionPolicy Bypass -File setup.ps1")
    return 0


if __name__ == "__main__":
    sys.exit(main())
