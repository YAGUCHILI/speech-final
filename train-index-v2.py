import os
import sys
import time
import numpy as np
import faiss
from tqdm import tqdm
import argparse

# ============ 命令行参数 ============

parser = argparse.ArgumentParser(description="构建FAISS索引")
parser.add_argument("--inp_root", type=str, required=True, help="特征目录")
parser.add_argument("--output_dir", type=str, required=True, help="索引输出目录")
parser.add_argument("--feature_dim", type=int, default=768, help="特征维度")
parser.add_argument("--index_prefix", type=str, default="added_index", help="索引前缀")
parser.add_argument("--force_gpu", action="store_true", help="强制使用GPU")
args = parser.parse_args()

inp_root = args.inp_root
output_dir = args.output_dir
feature_dim = args.feature_dim
index_prefix = args.index_prefix

print("=" * 50)
print("配置信息:")
print(f"  特征目录: {inp_root}")
print(f"  输出目录: {output_dir}")
print(f"  特征维度: {feature_dim}")
print(f"  索引前缀: {index_prefix}")
print(f"  强制GPU: {args.force_gpu}")
print("=" * 50)

# 检查输入目录是否存在
if not os.path.exists(inp_root):
    print(f"错误: 特征目录不存在: {inp_root}")
    sys.exit(1)

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# ============ 计算特征数量 ============

total_features = 0
for name in os.listdir(inp_root):
    if name.endswith('.npy'):
        phone = np.load(os.path.join(inp_root, name), mmap_mode='r')
        total_features += phone.shape[0]

print(f"总特征数: {total_features:,}")

# ============ 强制使用 Flat 索引（禁用 IVF） ============
print("\n" + "=" * 30)
print("使用 Flat 索引（已禁用 IVF）")
print("=" * 30)

# 创建 Flat 索引（使用内积，适合余弦相似度）
cpu_index = faiss.IndexFlatIP(feature_dim)

# ============ 尝试使用 GPU ============
use_gpu = False
try:
    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        # 设置临时内存
        res.setTempMemory(512 * 1024 * 1024)
        
        # 配置 GPU 索引
        gpu_config = faiss.GpuIndexFlatConfig()
        gpu_config.device = 0
        gpu_config.useFloat16 = True
        
        # 转换为 GPU 索引
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index, gpu_config)
        print(f"✓ 使用 GPU 加速 (设备数: {faiss.get_num_gpus()})")
        use_gpu = True
    else:
        raise RuntimeError("No GPU detected")
except Exception as e:
    print(f"使用 CPU 模式: {e}")
    index = cpu_index
    faiss.omp_set_num_threads(16)
    use_gpu = False

# ============ 加载所有特征 ============
print(f"\n加载特征文件...")
load_start = time.time()

all_features = []
feature_files = sorted([f for f in os.listdir(inp_root) if f.endswith('.npy')])
print(f"找到 {len(feature_files)} 个特征文件")

for name in tqdm(feature_files, desc="加载特征"):
    phone = np.load(os.path.join(inp_root, name))
    all_features.append(phone)
    print(f"  加载 {name}: {phone.shape}")

big_npy = np.concatenate(all_features, 0).astype(np.float32)
print(f"\n总特征数组: {big_npy.shape}")
print(f"内存占用: {big_npy.nbytes / 1024**3:.2f} GB")
print(f"加载耗时: {time.time() - load_start:.2f}秒")

# ============ 添加特征到索引 ============
print(f"\n添加特征到索引...")
add_start = time.time()

# 分批添加，避免内存问题
batch_size = 50000
n_batches = (big_npy.shape[0] + batch_size - 1) // batch_size
print(f"分批添加，批次大小: {batch_size:,}, 总批次: {n_batches}")

for i in range(0, big_npy.shape[0], batch_size):
    batch = big_npy[i:i+batch_size]
    index.add(batch)
    print(f"  已添加 {min(i+batch_size, big_npy.shape[0]):,}/{big_npy.shape[0]:,} 个特征")

print(f"添加完成，耗时: {time.time() - add_start:.2f}秒")
print(f"索引中向量数: {index.ntotal}")

# ============ 保存索引 ============
print("\n保存索引...")
save_start = time.time()

# 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)
final_path = os.path.join(output_dir, f"{index_prefix}.index")

# GPU 索引需要先转回 CPU 才能保存
if use_gpu:
    print("将 GPU 索引转换为 CPU 索引...")
    cpu_index_save = faiss.index_gpu_to_cpu(index)
    faiss.write_index(cpu_index_save, final_path)
else:
    faiss.write_index(index, final_path)

print(f"索引保存: {final_path}")
print(f"文件大小: {os.path.getsize(final_path) / 1024**3:.2f} GB")
print(f"保存耗时: {time.time() - save_start:.2f}秒")

# ============ 验证索引 ============
print("\n验证索引...")
test_vec = big_npy[:1]
D, I = index.search(test_vec, 5)
print(f"✓ 查询成功，最近距离: {D[0][0]:.4f}")
print(f"  找到的索引: {I[0]}")

# ============ 完成 ============
total_time = time.time() - load_start
print(f"\n总耗时: {total_time:.2f}秒")
print("=" * 50)
print("索引构建完成！")
print("=" * 50)