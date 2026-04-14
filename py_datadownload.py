import os
import requests
from datetime import datetime

# 基础 URL（根据你提供的链接）
base_url = "https://climserv.ipsl.polytechnique.fr/cfmip-obs/data/GOCCP_v3/3D_CloudFraction/grid_2x2xL40/2018/avg/"

# 文件名模板
# 第一种：3D_CloudFraction330m_YYYYMM_avg_CFMIP2_sat_3.1.2.nc
# 第二种：3D_CloudFraction_Phase330m_YYYYMM_avg_CFMIP2_sat_3.1.2.nc

# 创建保存目录
save_dir = "downloaded_nc"
os.makedirs(save_dir, exist_ok=True)

# 生成2018年1月到12月的月份列表
months = [f"2018{str(m).zfill(2)}" for m in range(2, 13)]

# 两种文件名前缀
prefixes = [
    "3D_CloudFraction330m",
    "3D_CloudFraction_Phase330m"
]

# 下载函数
def download_file(url, local_path):
    try:
        print(f"正在下载: {url}")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()  # 检查HTTP错误
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=262144):
                f.write(chunk)
        print(f"保存成功: {local_path}")
    except Exception as e:
        print(f"下载失败 {url}: {e}")

# 遍历所有月份和前缀
for month in months:
    for prefix in prefixes:
        filename = f"{prefix}_{month}_avg_CFMIP2_sat_3.1.2.nc"
        file_url = base_url + filename
        local_file = os.path.join(save_dir, filename)
        # 如果文件已存在，跳过（可选）
        if not os.path.exists(local_file):
            download_file(file_url, local_file)
        else:
            print(f"文件已存在，跳过: {local_file}")