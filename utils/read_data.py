import numpy as np
import os

# Data 폴더 경로
DATA_DIR = "/Users/pmj0324/Sicence/cosmo/Data"

print("=" * 80)
print("2D 데이터 분석")
print("=" * 80)

# 2D 데이터 읽기
data_2d_dir = os.path.join(DATA_DIR, "2D")

# params 파일 읽기
params_file = os.path.join(data_2d_dir, "params_LH_IllustrisTNG.txt")
print(f"\n파일: {params_file}")
with open(params_file, 'r') as f:
    lines = f.readlines()
    print(f"총 라인 수: {len(lines)}")
    print(f"처음 5줄:")
    for line in lines[:5]:
        print(f"  {line.strip()}")

# 2D Maps 읽기
maps_file = os.path.join(data_2d_dir, "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy")
print(f"\n파일: {maps_file}")
maps_2d = np.load(maps_file)
print(f"Shape: {maps_2d.shape}")
print(f"Data type: {maps_2d.dtype}")
print(f"Min value: {maps_2d.min():.6f}")
print(f"Max value: {maps_2d.max():.6f}")
print(f"Mean value: {maps_2d.mean():.6f}")
print(f"Std value: {maps_2d.std():.6f}")

print("\n" + "=" * 80)
print("3D 데이터 분석")
print("=" * 80)

# 3D 데이터 읽기
data_3d_dir = os.path.join(DATA_DIR, "3D")
files_3d = sorted([f for f in os.listdir(data_3d_dir) if f.endswith('.npy')])

print(f"\n3D 데이터 파일 목록: {len(files_3d)}개")
for file in files_3d:
    print(f"  - {file}")

# 각 파일의 정보 출력
print("\n" + "-" * 80)
for file in files_3d:
    file_path = os.path.join(data_3d_dir, file)
    print(f"\n파일: {file}")
    data = np.load(file_path)
    print(f"  Shape: {data.shape}")
    print(f"  Data type: {data.dtype}")
    print(f"  Min value: {data.min():.6f}")
    print(f"  Max value: {data.max():.6f}")
    print(f"  Mean value: {data.mean():.6f}")
    print(f"  Std value: {data.std():.6f}")
    print(f"  메모리 크기: {data.nbytes / (1024**2):.2f} MB")

