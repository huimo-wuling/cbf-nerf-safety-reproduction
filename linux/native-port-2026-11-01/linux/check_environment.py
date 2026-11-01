from pathlib import Path
import sys
import torch
import cv2
import open3d
import skimage
import trimesh
import colorama
root = Path(__file__).resolve().parents[1]
print('Python:', sys.version)
print('PyTorch:', torch.__version__)
print('CUDA:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit('GPU unavailable: run from a normal host terminal and check nvidia-smi')
print('GPU:', torch.cuda.get_device_name(0))
x = torch.ones(16, device='cuda'); print('CUDA computation:', (x*x).sum().item())
for file in ('replica_room1.npy', 'replica_room1.pth', 'pretrained/middle_fine.pt','Datasets/Replica/room1/traj.txt'):
    assert (root/file).is_file(), file
print('Environment and pretrained assets OK. Placeholder data, if used, is NOT training data.')
