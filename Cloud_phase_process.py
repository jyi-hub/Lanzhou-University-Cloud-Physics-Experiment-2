import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import xarray as xr
import numpy as np
import re
import glob


def read_data(file_path):
    """读取nc文件并返回xarray.Dataset对象
    args：
        file_path: nc文件路径
    returns：
        xarray.Dataset对象
    """
    #先丢弃

    #再根据文件名，补时间坐标
    try:
        ds = xr.open_dataset(file_path)
        print("普通打开成功")
    except ValueError as e:
        if "dimension 'time' already exists as a scalar variable" in str(e):
            print("检测到 time 元数据冲突，使用兼容模式打开")
            ds = xr.open_dataset(file_path, drop_variables=['time'])
            match = re.search(r"(\d{6})", file_path)#找到连续的6个数字
            if match and 'time' in ds.sizes and ds.sizes['time'] == 1:
                ym = match.group(1)
                ds = ds.assign_coords(time=('time', [np.datetime64(f"{ym[:4]}-{ym[4:]}-15")]))#增添（补上）时间坐标变量（坐标本质上也是变量，是特殊的变量）
        else:
            raise
    return ds
# file_path = r"云相态2018/3D_CloudFraction_Phase330m_201806_avg_CFMIP2_sat_3.1.2.nc"
# ds=read_data(file_path)
# ds



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
        
def cal_cld_xiangtai(ds):
    """计算云相态水平分布数据，返回绘图关键变量
    args：
        ds: xarray.Dataset对象
    returns：
        cld_ice: 云冰相态空间分布（删去时间维度）
        cld_liq: 云水相态空间分布（删去时间维度）
        cld_un: 云不确定相态空间分布（删去时间维度）
    """

    cld_ice = ds['clcalipso_ice'].mean(dim='altitude')
    cld_ice = cld_ice.isel(time=0, drop=True)
    cld_liq = ds['clcalipso_liq'].mean(dim='altitude')
    cld_liq = cld_liq.isel(time=0, drop=True)
    
    
    datas={'ice':cld_ice,'liq':cld_liq}

    return datas
# datas= cal_cld_xiangtai(ds)
# # print(cld_liq)
# # print(cld_liq.max())


def plt_xiangtai(lon, lat,datas, save_dir=None):
    """绘制云相态空间分布图
    args：
        lon: 经度坐标
        lat: 纬度坐标
        datas: 包含云相态绘图数据的字典，键为'ice'和'liq'
        save_dir: 图片保存目录，若为None则不保存

        cld_liq: 云水相态空间分布（删去时间维度）
        save_dir: 图片保存目录，若为None则不保存
    """

    #水平分布绘图
    cmap = 'jet'
    bounds = np.linspace(0, 1, 13)
    norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=256)

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    

    for key,data in datas.items():
        fig,axes=plt.subplots(1,2,figsize=(12,6),constrained_layout=True, subplot_kw={'projection': ccrs.Robinson()})
        ax1=axes[0].pcolormesh(lon, lat, data, cmap=cmap, shading='auto',norm=norm,transform=ccrs.PlateCarree())
        axes[1].remove()  # 移除第二个子图
        axes[1] = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
        axes[1].pcolormesh(lon, lat, data, cmap=cmap, shading='auto',norm=norm,transform=ccrs.PlateCarree())
        axes[0].set_title(f'global cloud {key} distribution',fontsize=16)
        axes[0].coastlines()
        axes[0].add_feature(cfeature.LAND, linewidth=0.5)
        axes[0].add_feature(cfeature.BORDERS, linewidth=0.5)
        axes[1].set_title(f'global cloud_{key} distribution',fontsize=16)
        axes[1].coastlines()
        axes[1].add_feature(cfeature.LAND, linewidth=0.5)
        axes[1].add_feature(cfeature.BORDERS, linewidth=0.5)
        cbar=plt.colorbar(ax1,ax=axes, orientation='horizontal',fraction=0.05, pad=0.1, aspect=40, label='色标')
        cbar.set_ticks(bounds,)
        cbar.set_label('色标', fontsize=12)
        
        #保存图像
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'global_cloud_{key}_distribution.png')
            fig.savefig(save_path)
            print(f'已保存: {save_path}')
        else:
            print('未指定保存目录，未保存图像。')
        plt.close(fig)

#plt_xiangtai(ds['longitude'], ds['latitude'], datas, save_dir=r"云相态2018\processed_images\2018-04-15")



def cal_vertical_xiangtai(ds):
    """计算云相态垂直分布
    args：
        ds: xarray.Dataset, 包含云相态数据
    returns：
        datasets: dict, 包含冰云和液态云等的垂直分布
    """
    v_cld_ice = ds['clcalipso_ice'].mean(dim='longitude')
    v_cld_liq = ds['clcalipso_liq'].mean(dim='longitude')
    v_cld_ice = v_cld_ice.isel(time=0, drop=True)
    v_cld_liq = v_cld_liq.isel(time=0, drop=True)
    datasets = {'ice_vertical': v_cld_ice, 'liq_vertical': v_cld_liq}
    return datasets

#cal_vertical_xiangtai(ds)


def plot_vertical_cld_distribution(lat,alt_mid,datasets,save_dir=None):
    """绘制云相态垂直分布图
    args:
        lat: 纬度坐标
        alt_mid: 云层中点高度
        datasets: dict, 包含冰云和液态云等的垂直分布
        save_dir: str, 图片保存目录
    """

    #设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    #cmap，bounds，norm等参数
    cmap = 'jet'
    bounds = [np.linspace(0, 0.3, 13), np.linspace(0, 1, 13)]
    norm = [
        mcolors.BoundaryNorm(boundaries=bounds[0], ncolors=256),
        mcolors.BoundaryNorm(boundaries=bounds[1], ncolors=256),
    ]


    #绘图循环
    for key,data in datasets.items():
        fig, ax = plt.subplots(figsize=(5, 4), constrained_layout=True)
        # if key != 'liq_vertical':
        #     cf1 = ax.pcolormesh(ds['latitude'], ds['alt_mid'], np.asarray(data), cmap=cmap, shading='auto', norm=norm[0])
        # else:
        #     cf1 = ax.pcolormesh(ds['latitude'], ds['alt_mid'], np.asarray(data), cmap=cmap, shading='auto', norm=norm[1])
        cf1=ax.pcolormesh(lat, alt_mid, data, cmap=cmap, shading='auto', norm=norm[0])
        ax.set_title(f'{key} distribution')
        ax.set_xlabel('Latitude')
        ax.set_ylabel('Altitude(KM)')
        fig.colorbar(cf1, ax=ax, orientation='vertical')
        #保存图像
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{key}_distribution.png')
            fig.savefig(save_path, dpi=300)
            print(f'已保存: {save_path}')
            
        else:
            print('未指定保存目录，未保存图像。')
        plt.close()

# save_dir=r"云相态2018\processed_images\2018-04-15"
# plot_vertical_cld_distribution(ds['latitude'], ds['alt_mid'], datasets, save_dir=save_dir)



def single_file_process_xiangtai(file_path, save_dir=None):
    """处理单个文件的云相态空间分布，包含计算和绘图。
    args：
        file_path: nc文件路径
        save_dir: 图片保存基本目录，若为None则不保存
    """

    ds = read_data(file_path)
    time_str = get_time_str(ds)

    if save_dir is not None:
        save_dir = os.path.join(save_dir, time_str)
        os.makedirs(save_dir, exist_ok=True)
    print(f"保存目录: {save_dir}")

    datas = cal_cld_xiangtai(ds)
    v_data = cal_vertical_xiangtai(ds)
    plt_xiangtai(ds['longitude'], ds['latitude'], datas, save_dir=save_dir)
    plot_vertical_cld_distribution(ds['latitude'], ds['alt_mid'], v_data, save_dir=save_dir)
file_path = r"云相态2018/3D_CloudFraction_Phase330m_201806_avg_CFMIP2_sat_3.1.2.nc" 
single_file_process_xiangtai(file_path, save_dir=r"云相态2018\processed_images")


def multi_file_process_xiangtai(folder_path, save_dir=None):
    """批量处理指定文件夹下的所有 NetCDF 文件的云相态空间分布。
    args：
        folder_path: 包含nc文件的文件夹路径
        save_dir: 图片保存基本目录，若为None则不保存
    """
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
            single_file_process_xiangtai(file_path, save_dir=save_dir)
        except Exception as e:
            print(f"处理失败: {file_path}\n错误信息: {e}")
            continue

    print("\n批量处理完成！")

#multi_file_process_xiangtai(r"云相态2018\新建文件夹", save_dir=r"云相态2018\processed_images")