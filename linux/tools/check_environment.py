# -*- coding: utf-8 -*-
"""Linux 复现环境自检。

用法(在仓库根目录):
    python linux/tools/check_environment.py

检查项:Python / PyTorch / CUDA 可用性 / 显卡型号 / 关键依赖 /
作者发布的神经隐式地图权重 / 数据集目录。任一项缺失会明确报错。
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
UPSTREAM = REPO / "upstream"

print("Python :", sys.version.split()[0], f"({sys.executable})")

ok = True


def check(label, fn):
    global ok
    try:
        print(f"{label:9s}:", fn())
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"{label:9s}: 失败 -> {exc}")


import torch  # noqa: E402

print("PyTorch :", torch.__version__)
print("CUDA    :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU     :", torch.cuda.get_device_name(0))
    props = torch.cuda.get_device_properties(0)
    print("显存    :", f"{props.total_memory / 2**30:.1f} GiB")
    x = torch.ones(1024, device="cuda")
    print("实测计算:", f"{(x * x).sum().item():.0f}")

check("cv2", lambda: __import__("cv2").__version__)
check("open3d", lambda: __import__("open3d").__version__)
check("skimage", lambda: __import__("skimage").__version__)
check("trimesh", lambda: __import__("trimesh").__version__)
check("scipy", lambda: __import__("scipy").__version__)
check("matplotlib", lambda: __import__("matplotlib").__version__)

for rel in (
    "replica_room1.npy",
    "replica_room1.pth",
    "pretrained/middle_fine.pt",
    "pretrained/coarse.pt",
    "configs/Replica/room1.yaml",
    "single_cbf.py",
    "double_cbf.py",
):
    p = UPSTREAM / rel
    if p.is_file():
        print(f"asset    : ok  {rel}  ({p.stat().st_size / 2**20:.1f} MiB)")
    else:
        ok = False
        print(f"asset    : 缺失 {rel}")

dataset = UPSTREAM / "Datasets/Replica/room1/traj.txt"
print(f"dataset  : {'ok  ' + str(dataset) if dataset.is_file() else '未就绪(可用 tools/make_dataset_stub.py 生成 1 帧占位)'}")

if not torch.cuda.is_available():
    ok = False
    print("\n[!] CUDA 不可用。请在普通宿主终端(非容器/非沙箱)运行,并确认 `nvidia-smi` 正常。")

print("\n结论:", "环境就绪 ✓" if ok else "环境有问题 ✗")
sys.exit(0 if ok else 1)
