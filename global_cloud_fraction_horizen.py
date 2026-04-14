import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import os
import glob
import re


def get_time_str(ds):
    """兼容标量/数组 time 坐标，返回 YYYY-MM-DD 形式日期字符串。"""
    time_var = ds.get('time')
    if time_var is None:
        return 'unknown_time'

    try:
        time_value = np.asarray(time_var.values).reshape(-1)[0]
        if np.isnat(time_value):
            return 'unknown_time'
        return np.datetime_as_string(time_value, unit='D')
    except Exception:
        try:
            return str(np.asarray(time_var.values).reshape(-1)[0])[:10]
        except Exception:
            return 'unknown_time'


def open_dataset_safe(file_path):
    """安全打开数据集，兼容 time 维度与同名标量变量冲突。"""
    try:
        return xr.open_dataset(file_path)
    except ValueError as e:
        if "dimension 'time' already exists as a scalar variable" not in str(e):
            raise

        ds = xr.open_dataset(file_path, drop_variables=['time'])
        match = re.search(r'(\d{6})', os.path.basename(file_path))
        if match and 'time' in ds.dims and ds.dims['time'] == 1:
            ym = match.group(1)
            time_value = np.datetime64(f"{ym[:4]}-{ym[4:]}-15")
            ds = ds.assign_coords(time=('time', [time_value]))
        print(f"检测到 time 元数据冲突，已自动兼容处理: {os.path.basename(file_path)}")
        return ds




def calc_cloud(ds):
    """
    基于云上边界，区分低中高，并计算各自云量
    参数说明：
    输入：ds（xarray.Dataset），需包含'clcalipso'和'alt_bound'变量
    返回：datasets (list of (title, data) tuples，包含低云、中云、高云、总云量)
    """

    up_bound = ds['alt_bound'][1].values
    low_idx = (up_bound < 2)
    mid_idx = (up_bound >= 2) & (up_bound < 7)
    high_idx = (up_bound >= 7)
    total_idx = (up_bound >= 0)
    cld_low = ds['clcalipso'].isel(altitude=low_idx).mean(dim='altitude')
    cld_mid = ds['clcalipso'].isel(altitude=mid_idx).mean(dim='altitude')
    cld_high = ds['clcalipso'].isel(altitude=high_idx).mean(dim='altitude')
    cld_total = ds['clcalipso'].isel(altitude=total_idx).mean(dim='altitude')
    if 'time' in cld_low.dims:
        cld_low = cld_low.isel(time=0, drop=True)
    if 'time' in cld_mid.dims:
        cld_mid = cld_mid.isel(time=0, drop=True)
    if 'time' in cld_high.dims:
        cld_high = cld_high.isel(time=0, drop=True)
    if 'time' in cld_total.dims:
        cld_total = cld_total.isel(time=0, drop=True)
    
    return [
        ('Low Cloud Fraction (0-2 km)', cld_low),
        ('Mid Cloud Fraction (2-7 km)', cld_mid),
        ('High Cloud Fraction (>=7 km)', cld_high),
        ('Total Cloud Fraction', cld_total),
    ]

def global_cloud_fraction_layout(lon, lat, datasets, cmap='jet', vmin=0, vmax=1,save_dir=None):
    """
    全球云量水平分布布局函数，支持多个云层数据的可视化。
    参数说明：
    datasets: list of (title, data) tuples.
    eg:
    cloud_datasets = [
    ('Low Cloud Fraction (0-2 km)', cld_low),
    ('Mid Cloud Fraction (2-7 km)', cld_mid),
    ('High Cloud Fraction (>=7 km)', cld_high),
    ('Total Cloud Fraction', cld_total),
    ]

    lon,lat:df1['longitude'], df1['latitude']
    cmap:颜色映射，vmin/vmax:色标范围

    返回：figs (list of (fig, axes, cbar) tuples)
    """


    #绘图与保存
    #figs = []
    bounds = np.linspace(vmin, vmax, 9)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    for title, data in datasets:
        fig = plt.figure(figsize=(14, 6), constrained_layout=True)
        ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
        ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.Robinson())

        cf1 = ax1.pcolormesh(lon, lat, np.asarray(data), cmap=cmap, shading='auto', transform=ccrs.PlateCarree(), norm=norm)
        ax1.coastlines()
        ax1.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax1.set_title(f'{title} - PlateCarree')
        ax1.set_xlabel('Longitude')
        ax1.set_ylabel('Latitude')

        cf2 = ax2.pcolormesh(lon, lat, np.asarray(data), cmap=cmap, shading='auto', transform=ccrs.PlateCarree(), norm=norm)
        ax2.coastlines()
        ax2.add_feature(cfeature.BORDERS, linewidth=0.5)
        ax2.set_title(f'{title} - Robinson')

        cbar = fig.colorbar(cf2, ax=[ax1, ax2], orientation='horizontal', fraction=0.07, pad=0.2,
                            aspect=50, label='Cloud Fraction')
        #figs.append((fig, (ax1, ax2), cbar))
        

        # 自动保存
        if save_dir is not None:
            # 文件名中去除空格和特殊字符
            safe_title = title.replace(' ', '_').replace('(', '').replace(')', '').replace('>=', 'ge').replace('-', '').replace(',', '')
            save_path = os.path.join(save_dir, f'{safe_title}.png')
            fig.savefig(save_path, dpi=300)
            print(f'已保存: {save_path}')
        plt.close(fig)

    #return figs

def single_file_process_horizontal(ds,save_dir=None):
    """
    总体处理函数，包含云层计算、绘图与保存
    参数说明：
    输入：ds（xarray.Dataset），需包含'clcalipso'和'alt_bound'变量
    save_dir: 图片保存基本目录，若为None则不保存
    """

    # 按时间创建文件夹，兼容 time 为标量坐标的情况
    time_str = get_time_str(ds)

    # 构建保存目录
    if save_dir is not None:
        save_dir = os.path.join(save_dir,time_str) 
        os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录: {save_dir}")

    #计算各云层数据
    cloud_datasets = calc_cloud(ds)
    #绘图与保存
    #figs=global_cloud_fraction_layout(ds['longitude'], ds['latitude'], cloud_datasets, save_dir=save_dir)
    global_cloud_fraction_layout(ds['longitude'], ds['latitude'], cloud_datasets, save_dir=save_dir)



#以上为云量水平分布的计算与绘图函数，以下为云量垂直分布的计算与绘图函数

def cal_cld_vertical(ds):
    """计算全球云量的垂直分布：cloud、clear、uncalipso。

    Args:
        ds (xarray.Dataset): 包含'clcalipso'、'clrcalipso'和'uncalipso'变量的xarray数据集
    
    Returns:
        list of (title, data) tuples: 包含'cloud'、'clear'和'uncalipso'的垂直分布数据
    """
    Cloud = ds['clcalipso'].mean(dim=['longitude'])#CALIPSO 3D Cloud fraction ;
    Clear = ds['clrcalipso'].mean(dim=['longitude'])  #CALIPSO 3D Clear fraction ;
    uncalipso = ds['uncalipso'].mean(dim=['longitude'])  #CALIPSO 3D Undefined fraction ;
    if 'time' in Cloud.dims:
        Cloud = Cloud.isel(time=0, drop=True)
    if 'time' in Clear.dims:
        Clear = Clear.isel(time=0, drop=True)
    if 'time' in uncalipso.dims:
        uncalipso = uncalipso.isel(time=0, drop=True)

    return [
        ('Cloud Fraction Vertical Distribution', Cloud),
        ('Clear Fraction Vertical Distribution', Clear),
        ('Undefined Fraction Vertical Distribution', uncalipso),
       
    ]

def plot_vertical_distribution(lat, alt,datasets,save_dir=None):
    """绘制云量的垂直分布图。

    args:
        latitude (xarray.DataArray): 包含纬度信息的xarray数据数组
        altitude ,给实参的时候给表示云层中心高度的 ds['alt_mid']
        datasets (list of (title, data) tuples): 包含标题和数据的列表
    returns:
        None
    """
    cmap='jet'
    bounds = [np.linspace(0, 0.5, 13),np.linspace(0,1,13)]
    norm = [mcolors.BoundaryNorm(boundaries=bounds[0], ncolors=256),
            mcolors.BoundaryNorm(boundaries=bounds[1], ncolors=256)]

    for idx,(title, data) in enumerate(datasets):

        fig,ax = plt.subplots(figsize=(5,4), constrained_layout=True)
        if idx != 1:
            cf1 = ax.pcolormesh(lat, alt, np.asarray(data), cmap=cmap, shading='auto', norm=norm[0])
        else:
            cf1 = ax.pcolormesh(lat, alt, np.asarray(data), cmap=cmap, shading='auto', norm=norm[1])
        ax.set_title(f'{title}')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Altitude(KM)')

        fig.colorbar(cf1, ax=ax, orientation='vertical'
                            )
        
        # fig, ax = plt.subplots(figsize=(7, 8))
        # levels = np.linspace(0, 1, 21)

        # cf = ax.contourf(
        #     df1["latitude"],      # X
        #     df1["alt_mid"],       # Y
        #     data,                # Z (2D)
        #     levels=levels,
        #     cmap="jet",
        #     extend="both"
        # )

        # ax.set_xlabel("Latitude")
        # ax.set_ylabel("Altitude (km)")
        # ax.set_title("Cloud Fraction Vertical Distribution")
        # plt.colorbar(cf, ax=ax, label="Cloud Fraction")
        

        # 自动保存
        if save_dir is not None:
            # 文件名中去除空格和特殊字符
            save_path = os.path.join(save_dir, f'{title}.png')
            fig.savefig(save_path, dpi=300)
            print(f'已保存: {save_path}')
        else:
            print('未指定保存目录，未保存图像。')
        plt.close(fig)

def single_file_process_vertical(ds,save_dir=None):
    """
    处理单个文件的云量垂直分布，包含计算和绘图。

    参数说明：
    输入：ds（xarray.Dataset），需包含'clcalipso'、'clrcalipso'和'uncalipso'变量
    save_dir: 图片保存基本目录，若为None则不保存

    returns: None
    """

    # 按时间创建文件夹，兼容 time 为标量坐标的情况
    time_str = get_time_str(ds)

    # 构建保存目录
    if save_dir is not None:
        save_dir = os.path.join(save_dir,time_str) 
        os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录: {save_dir}")

    #计算云量垂直分布数据
    vertical_datasets = cal_cld_vertical(ds)
    
    #绘图与保存
    plot_vertical_distribution(ds['latitude'], ds['alt_mid'], vertical_datasets, save_dir=save_dir)

def multi_file_process(folder_path, save_dir=None):
    """
    批量处理指定文件夹下的所有 NetCDF 文件。绘制水平、垂直分布图，并保存到指定目录（如果 save_dir 不为 None）。
    参数说明：
        folder_path: str, 存放 nc 文件的文件夹路径
        save_dir: 图片保存基本目录，若为 None 则不保存图片
    结果：保存处理后的图片到指定目录（如果 save_dir 不为 None），并在控制台输出处理进度和结果。
    """
    # 获取文件夹下所有 .nc 文件
    file_pattern = os.path.join(folder_path, "*.nc")
    file_paths = glob.glob(file_pattern)
    
    if not file_paths:
        print(f"文件夹 {folder_path} 中没有找到任何 .nc 文件")
        return
    
    total = len(file_paths)
    print(f"找到 {total} 个 nc 文件，开始批量处理...")
    
    for idx, file_path in enumerate(file_paths, start=1):
        print(f"\n[{idx}/{total}] 正在处理: {os.path.basename(file_path)}")
        try:
            with open_dataset_safe(file_path) as ds:
                single_file_process_horizontal(ds, save_dir=save_dir)
                single_file_process_vertical(ds, save_dir=save_dir)
        except Exception as e:
            print(f"处理失败: {file_path}\n错误信息: {e}")
            continue
    
    print("\n批量处理完成！")

def main():
    folder_path = r"test云量垂直"
    save_dir = r"云量18年\processed_images"
    multi_file_process(folder_path, save_dir)

if __name__ == "__main__":
    main()
