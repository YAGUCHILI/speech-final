"""
针对小数据量的优化版本（4.3万特征）
"""
import os
import time
import numpy as np
import faiss
from tqdm import tqdm

# ============ 配置 ============
inp_root = r"./logs/my_models3/3_feature768"
output_dir = r"./logs/my_models3"
feature_dim = 768

# 根据数据量自动调整参数
total_features = 0
for name in os.listdir(inp_root):
    if name.endswith('.npy'):
        phone = np.load(f"{inp_root}/{name}", mmap_mode='r')
        total_features += phone.shape[0]

print(f"总特征数: {total_features:,}")

# 自动调整参数（数据量小时用全部数据）
if total_features < 100000:
    # 小数据量：用全部数据，不需要采样
    SAMPLE_SIZE = total_features  # 使用全部数据
    BATCH_SIZE_ADD = total_features  # 一次性添加
    USE_IVF = False  # 小数据量不需要IVF
    print("数据量小，使用Flat索引（最快）")
else:
    # 大数据量：使用优化参数
    SAMPLE_SIZE = min(500000, total_features // 2)
    BATCH_SIZE_ADD = min(300000, total_features // 10)
    USE_IVF = True

# ============ 构建索引 ============
if not USE_IVF or total_features < 100000:
    # 使用最简单的Flat索引（最快，最准确）
    print("使用IndexFlatL2索引...")
    index = faiss.IndexFlatL2(feature_dim)
    print("无需训练，直接添加")
else:
    # 使用IVF索引
    n_ivf = min(int(16 * np.sqrt(total_features)), total_features // 39)
    n_ivf = max(n_ivf, 100)
    print(f"使用IVF{n_ivf}索引...")
    
    index_desc = f"IVF{n_ivf},Flat"
    cpu_index = faiss.index_factory(feature_dim, index_desc)
    
    # 尝试GPU（如果可用）
    try:
        res = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = 0
        config.useFloat16 = True
        index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        print("使用GPU加速")
    except:
        index = cpu_index
        print("使用CPU")
        faiss.omp_set_num_threads(16)
    
    # 训练
    print(f"训练索引 (使用{SAMPLE_SIZE:,}个样本)...")
    train_data = []
    listdir_res = sorted(os.listdir(inp_root))
    remaining = SAMPLE_SIZE
    
    for name in listdir_res:
        if remaining <= 0:
            break
        phone = np.load(f"{inp_root}/{name}", mmap_mode='r')
        take = min(phone.shape[0], remaining)
        if take == phone.shape[0]:
            train_data.append(phone)
        else:
            indices = np.random.choice(phone.shape[0], take, replace=False)
            train_data.append(phone[indices])
        remaining -= take
    
    train_data = np.concatenate(train_data, 0).astype(np.float32)
    print(f"训练数据形状: {train_data.shape}")
    
    train_start = time.time()
    index.train(train_data)
    print(f"训练完成，耗时: {time.time()-train_start:.2f}秒")

# ============ 添加特征 ============
print(f"\n添加所有特征...")
add_start = time.time()

# 加载所有特征
all_features = []
for name in sorted(os.listdir(inp_root)):
    if name.endswith('.npy'):
        phone = np.load(f"{inp_root}/{name}")
        all_features.append(phone)
        print(f"加载 {name}: {phone.shape}")

big_npy = np.concatenate(all_features, 0).astype(np.float32)
print(f"总特征数组: {big_npy.shape}, 内存: {big_npy.nbytes/1024**3:.2f}GB")

# 添加（小数据量可以一次性添加）
index.add(big_npy)
print(f"添加完成，耗时: {time.time()-add_start:.2f}秒")
print(f"索引中向量数: {index.ntotal}")

# ============ 保存索引 ============
print("\n保存索引...")
save_start = time.time()

final_path = os.path.join(output_dir, f"added_index_{total_features}.index")
faiss.write_index(index, final_path)

print(f"索引保存: {final_path}")
print(f"文件大小: {os.path.getsize(final_path)/1024**3:.2f}GB")
print(f"保存耗时: {time.time()-save_start:.2f}秒")

# ============ 验证 ============
print("\n验证索引...")
test_vec = big_npy[:1]
D, I = index.search(test_vec, 5)
print(f"查询成功，最近距离: {D[0][0]:.4f}")
print(f"找到的索引: {I[0]}")

total_time = time.time() - add_start + (time.time()-save_start)
print(f"\n总耗时: {total_time:.2f}秒")