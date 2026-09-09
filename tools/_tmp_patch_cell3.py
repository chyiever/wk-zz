"""临时脚本：更新 build_sliding_window_notebook_v2_1.py 中的 Cell 3"""
import sys

path = 'tools/build_sliding_window_notebook_v2_1.py'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# Find Cell 3 start and Cell 4 start
cell3_marker = "    # Cell 3:"
cell4_marker = "    # Cell 4:"
idx3 = content.index(cell3_marker)
idx4 = content.index(cell4_marker)

new_cell3 = """    # Cell 3: 文件发现
    cells.append(make_cell('code', '''# =========================
# 数据文件发现
# =========================

source_files = discover_source_files(RAW_DATA_ROOTS, max_files=MAX_FILES)

if not source_files:
    raise FileNotFoundError(f'未找到任何 .npz/.tdms 文件，请检查路径: {RAW_DATA_ROOTS}')

from collections import Counter
folder_counts = Counter(f.parent.name for f in source_files)

print(f'发现 {len(source_files)} 个源文件（降采样前）')
print(f'\\\\n各文件夹文件数:')
for folder, count in sorted(folder_counts.items()):
    print(f'  {folder}: {count}')

npz_count = sum(1 for f in source_files if f.suffix.lower() == '.npz')
tdms_count = sum(1 for f in source_files if f.suffix.lower() == '.tdms')
print(f'\\\\n文件格式: {npz_count} npz, {tdms_count} tdms')

# 文件级降采样
if ENABLE_FILE_DOWNSAMPLING:
    source_files = downsample_source_files(
        source_files,
        ratio=FILE_SAMPLE_RATIO,
        seed=FILE_SAMPLE_SEED,
    )
    folder_counts_after = Counter(f.parent.name for f in source_files)
    print(f'\\\\n降采样后 {len(source_files)} 个源文件（比例={FILE_SAMPLE_RATIO}）')
    print(f'\\\\n各文件夹降采样后文件数:')
    for folder, count in sorted(folder_counts_after.items()):
        orig = folder_counts.get(folder, 0)
        print(f'  {folder}: {count} / {orig} ({count/orig*100:.1f}%)')
else:
    print(f'\\\\n未启用降采样，使用全部 {len(source_files)} 个文件')
    '''))

"""

content = content[:idx3] + new_cell3 + content[idx4:]

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print('SUCCESS: Cell 3 replaced')
