import os
import pyvips
from tqdm import tqdm
import ephem  
import math
from datetime import datetime, timedelta
import pandas as pd 
import numpy as np
import rasterio
from rasterio.transform import from_origin
from numba import cuda
import time
from rasterio.mask import mask
from rasterio.crs import CRS
from rasterio.windows import Window
import gc
import psutil
from rasterio.io import MemoryFile

# 文件路径
file_class = "DSM"
original_image = r'E:\research\shade_data\cities\Wenzhou_DSM_improve\Wenzhou_DSM_16bit.tif'
base_output_dir = r'E:\research\shade_data\cities\Wenzhou_DSM_improve'
shapefile = r"E:\research\shade_data\cities\Wenzhou_DSM_improve\Wenzhou.shp"
city = "Wenzhou"
start_hour = 18
end_hour = 18

# 使用D盘作为临时文件夹
temp_dir = os.path.join('E:', 'temp_shadow_processing')
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(base_output_dir, exist_ok=True)

def clear_memory():
    """清理内存和缓存"""
    # 同步CUDA设备，确保所有操作完成
    if cuda.is_available():
        try:
            cuda.synchronize()
            # 重置当前CUDA上下文
            ctx = cuda.current_context()
            ctx.reset()
            # 显式释放所有分配的内存
            ctx.memory_manager.deallocations.clear()
            ctx.memory_manager.allocations.clear()
        except Exception as e:
            print(f"CUDA清理出错: {str(e)}")
            # 如果出错，尝试重新创建上下文
            device = cuda.get_current_device()
            cuda.create_context(device)
    
    # 强制垃圾回收
    gc.collect()
    gc.collect()

# 经纬度与时间信息
latitude = 28.0000
longitude = 120.7511
date = '2024/06/21'
utc_offset = 8
height = 1.7
"""2024.6.21 日出:4:58 日落:19:04"""

def calculate_sun_position(latitude, longitude, local_date, utc_offset):
    local_time = datetime.strptime(local_date, '%Y/%m/%d %H:%M:%S')
    utc_time = local_time - timedelta(hours=utc_offset)

    obs = ephem.Observer()
    obs.lat, obs.lon = str(latitude), str(longitude) 
    obs.date = ephem.Date(utc_time.strftime('%Y/%m/%d %H:%M:%S')) 

    sun = ephem.Sun()
    sun.compute(obs)

    sun_altitude_deg = math.degrees(sun.alt)
    sun_azimuth_deg = math.degrees(sun.az)  # 方位角

    return sun_altitude_deg, sun_azimuth_deg

# rotate
def rotate_image(input_path, output_path, angle):
    print("正在旋转图像...")
    x = pyvips.Image.new_from_file(input_path)

    with tqdm(total=100, desc="旋转中") as pbar:
        rotated_image = x.rotate(angle, interpolate=pyvips.Interpolate.new("nearest"))
        pbar.update(100)  # 旋转完成后更新进度条

    with tqdm(total=100, desc="保存中") as pbar:
        rotated_image.write_to_file(output_path, compression="lzw")
        pbar.update(100)  # 保存完成后更新进度条
    print("旋转处理完成!")

# CUDA核心函数 - 计算地平线角度
@cuda.jit
def cuda_horizon_angle_kernel(dem_data, shadow_mask, sun_altitude_rad, height_value):
    """使用CUDA沿Y方向（南向）计算阴影"""
    row, col = cuda.grid(2)
    rows, cols = dem_data.shape
    
    if row < rows and col < cols:
        # 初始化为0
        shadow_mask[row, col] = 0
        
        # 如果是无效值，跳过计算
        if dem_data[row, col] <= 0:
            return
            
        current_elevation = dem_data[row, col]
        max_angle = -999.0
        is_shadowed = False
        
        # 搜索范围
        max_search = min(500, rows - row)
        
        # 计算阴影
        for dist in range(1, max_search):
            next_row = row + dist
            if next_row >= rows:
                break
                
            # 计算高度差，考虑高度值修正
            elevation_diff = dem_data[next_row, col] - current_elevation - height_value
            
            # 只有当高度差为正时才计算角度
            if elevation_diff > 0:
                # 计算角度
                angle = math.atan2(elevation_diff, float(dist))
                max_angle = max(angle, max_angle)
                
                # 如果角度大于太阳高度角，则在阴影中
                if angle > sun_altitude_rad:
                    is_shadowed = True
                    shadow_mask[row, col] = 1
                    break

# 针对内存较小的系统，使用分块处理的CUDA函数
def calculate_shade_cuda_chunked(input_path, output_path, sun_altitude_rad, height_value, chunk_size=1024):
    # # 确保CUDA上下文是新的或有效的
    # if cuda.is_available():
    #     try:
    #         # 尝试获取当前上下文，如果失败则创建新的
    #         ctx = cuda.current_context()
    #     except:
    #         # 创建新的CUDA上下文
    #         device = cuda.get_current_device()
    #         ctx = cuda.create_context(device)
    
    print("使用分块CUDA计算阴影...")
    start_time = time.time()
    
    # 读取输入数据的元数据
    with rasterio.open(input_path) as src:
        profile = src.profile
        total_rows, total_cols = src.shape
    
    print(f"总图像尺寸: {total_rows}x{total_cols}")
    
    # 创建输出文件
    profile.update(dtype=rasterio.uint8, count=1, compress='lzw')
    
    with rasterio.open(output_path, 'w', **profile) as dst:
        # 分块处理
        with tqdm(total=total_rows, desc="处理区块") as pbar:
            for row_start in range(0, total_rows, chunk_size):
                chunk_start_time = time.time()
                
                # 计算当前块的结束行
                row_end = min(row_start + chunk_size, total_rows)
                chunk_rows = row_end - row_start
                
                # 读取当前块
                with rasterio.open(input_path) as src:
                    window = ((row_start, row_end), (0, total_cols))
                    dem_chunk = src.read(1, window=window).astype(np.float32)
                
                # 创建该块的阴影掩码
                shadow_chunk = np.zeros((chunk_rows, total_cols), dtype=np.uint8)
                
                # 设置CUDA网格和块大小
                threads_per_block = (16, 16)
                blocks_x = (chunk_rows + threads_per_block[0] - 1) // threads_per_block[0]
                blocks_y = (total_cols + threads_per_block[1] - 1) // threads_per_block[1]
                blocks_per_grid = (blocks_x, blocks_y)
                  # 将数据复制到GPU
                try:
                    d_dem_chunk = cuda.to_device(dem_chunk)
                    d_shadow_chunk = cuda.to_device(shadow_chunk)
                    
                    # 执行CUDA核心函数
                    cuda_horizon_angle_kernel[blocks_per_grid, threads_per_block](
                        d_dem_chunk, d_shadow_chunk, sun_altitude_rad, height_value)
                    cuda.synchronize()
                    
                    # 将结果从GPU复制回主机
                    shadow_chunk = d_shadow_chunk.copy_to_host()
                    
                except Exception as e:
                    print(f"CUDA处理出错: {str(e)}")
                    # 如果CUDA出错，创建一个全零的阴影数组
                    shadow_chunk = np.zeros((chunk_rows, total_cols), dtype=np.uint8)
                    
                finally:
                    # 确保释放GPU内存
                    try:
                        if 'd_dem_chunk' in locals():
                            del d_dem_chunk
                        if 'd_shadow_chunk' in locals():
                            del d_shadow_chunk
                        cuda.synchronize()
                    except:
                        pass
                
                # 将结果写入输出文件
                dst.write(shadow_chunk, 1, window=window)
                
                # 更新进度条
                pbar.update(chunk_rows)
                
                chunk_end_time = time.time()
                print(f"区块 {row_start}-{row_end} 处理完成，用时: {chunk_end_time - chunk_start_time:.2f}秒")
                
    end_time = time.time()

    if os.path.exists(output_path):
        # 打印文件大小，单位MB
        print(f"文件存在。大小: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    else:
        print("⚠️ 严重警告：CUDA 输出文件不存在！")

    print(f"分块CUDA处理完成，总用时: {end_time - start_time:.2f}秒")
    print("阴影计算完成!")

def get_original_size_and_metadata(original_path):
    """获取原始影像的大小和元数据信息
    
    Args:
        original_path: 原始影像路径
    Returns:
        tuple: (rows, cols, transform, crs)
    """
    with rasterio.open(original_path) as src:
        return (src.height, src.width, src.transform, src.crs)

def crop_to_original_size(final_path, original_path, output_path, chunk_size=1024):
    """裁剪旋转后的影像，使其大小与原始影像一致，并保持地理信息
       同时按原始图像的 nodata 区域做掩膜，输出图像中 nodata 区域设置为0

    Args:
        final_path: 旋转后的影像路径
        original_path: 原始影像路径
        output_path: 输出路径
        chunk_size: 分块大小，默认1024
    """
    import numpy as np
    import rasterio
    from rasterio.windows import Window

    print("正在裁剪旋转后的影像...")

    # 获取原始影像大小和地理信息
    orig_rows, orig_cols, geo_transform, projection = get_original_size_and_metadata(original_path)
    print(f"原始影像尺寸: {orig_rows}x{orig_cols}")

    with rasterio.open(original_path) as orig_src, rasterio.open(final_path) as src:
        print(f"旋转后的影像尺寸: {src.height}x{src.width}")
        # 获取原始 nodata 掩膜
        with rasterio.open(original_path) as orig_src:
            orig_mask = np.ones((orig_src.height, orig_src.width), dtype=bool)
        
        # 计算旋转图的中心点，并据此裁剪出和原图同样大小的区域
        center_x, center_y = src.width // 2, src.height // 2
        start_x = center_x - (orig_cols // 2)
        start_y = center_y - (orig_rows // 2)

        # 限制在图像范围内
        start_x = max(start_x, 0)
        start_y = max(start_y, 0)
        end_x = min(start_x + orig_cols, src.width)
        end_y = min(start_y + orig_rows, src.height)

        # 最终裁剪尺寸（实际可能略小于 orig_rows/cols）
        crop_width = end_x - start_x
        crop_height = end_y - start_y

        # 输出图像 profile
        profile = src.profile.copy()
        profile.update({
            'height': crop_height,
            'width': crop_width,
            'transform': geo_transform,
            'crs': projection,
            'tiled': True,
            'blockxsize': chunk_size,
            'blockysize': chunk_size,
            'compress': 'deflate'
        })

        print(f"输出影像尺寸: {crop_height}x{crop_width}")

        with rasterio.open(output_path, 'w', **profile) as dst:
            for y in range(0, crop_height, chunk_size):
                h = min(chunk_size, crop_height - y)
                for x in range(0, crop_width, chunk_size):
                    w = min(chunk_size, crop_width - x)

                    window = Window(start_x + x, start_y + y, w, h)
                    data = src.read(window=window)

                    # 获取对应的原始掩膜块
                    mask_window = Window(x, y, w, h)
                    orig_valid_mask = orig_mask[y:y + h, x:x + w]

                    # 对应位置无效区域设置为0
                    for band in range(data.shape[0]):
                        data[band, ~orig_valid_mask] = 0

                    # 写入输出图像
                    dst.write(data, window=Window(x, y, w, h))

    print(f"✅ 已完成输出：{output_path}")


def get_memory_usage():
    """获取当前内存使用情况（MB）"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def process_hourly_shadows(start_hour, end_hour, use_chunked=True):
    """处理每小时的阴影，可选择是否使用分块处理"""
    # 检测CUDA是否可用
    if not cuda.is_available():
        print("⚠️ 当前系统不支持CUDA! 将回退到CPU处理。")
        use_cuda = False
    else:
        print(f"✅ CUDA可用。设备: {cuda.get_current_device().name}")
        use_cuda = True
    
    ## 全局时间，用真实时间
    total_start_time = time.time()
    
    # 创建临时文件夹
    temp_dir = os.path.join(base_output_dir, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        for hour in range(start_hour, end_hour + 1):
            # # 在每次循环开始时重新初始化CUDA环境
            # if cuda.is_available():
            #     try:
            #         # 尝试重置当前上下文
            #         cuda.current_context().reset()
            #     except:
            #         # 如果失败，创建新的上下文
            #         device = cuda.get_current_device()
            #         cuda.create_context(device)

            hour_start_time = time.time()
            time_str = f"{hour:02d}:00:00"
            local_date = f"{date} {time_str}"
            file_suffix = time_str.replace(':', '')

            # 使用临时文件夹存储中间文件
            rotated_image_path = os.path.join(temp_dir, f'{city}_{file_class}_rotate_{file_suffix}.tif')
            shadow_image_path = os.path.join(temp_dir, f'{city}_{file_class}_shadow_{file_suffix}.tif')
            final_image_path = os.path.join(temp_dir, f'{city}_{file_class}_shade_{file_suffix}_test.tif')
            output_masked_path = os.path.join(base_output_dir, f'{city}_{file_class}_shade_{file_suffix}.tif')

            # 获取太阳角度
            sun_altitude_angle, sun_azimuth_angle = calculate_sun_position(latitude, longitude, local_date, utc_offset)
            sun_altitude_rad = np.radians(sun_altitude_angle)
            rotate_angle = 180-sun_azimuth_angle
            recover_angle = -(180-sun_azimuth_angle)
            print(f"⛅ 正在处理 {time_str} - 太阳高度角: {sun_altitude_angle:.2f}°, 太阳方位角: {sun_azimuth_angle:.2f}°")
            
            # 旋转图像
            rotate_image(original_image, rotated_image_path, rotate_angle)
            
            # 计算阴影 - 使用CUDA或CPU
            if use_cuda:
                if use_chunked:
                    calculate_shade_cuda_chunked(rotated_image_path, shadow_image_path, sun_altitude_rad, height)

            # 反向旋转
            rotate_image(shadow_image_path, final_image_path, recover_angle)

            # 提取原始影像信息
            orig_rows, orig_cols, geo_transform, projection = get_original_size_and_metadata(original_image)

            # 裁剪旋转后的影像，并保持地理信息
            crop_to_original_size(final_image_path, original_image, output_masked_path)

            # finally:
            #     # 立即清理当前小时的中间文件
            #     for file in [rotated_image_path, shadow_image_path, final_image_path]:
            #         if os.path.exists(file):
            #             try:
            #                 os.remove(file)
            #                 print(f"已清理中间文件: {file}")
            #             except Exception as e:
            #                 print(f"清理文件 {file} 时出错: {str(e)}")
            
            # 再次清理内存
            clear_memory()
            
            print(f"✅ {time_str} 的阴影处理完成！")
    
    finally:
        # # 清理临时文件夹
        # try:
        #     import shutil
        #     if os.path.exists(temp_dir):
        #         shutil.rmtree(temp_dir)
        #         print(f"已清理临时文件夹: {temp_dir}")
        # except Exception as e:
        #     print(f"清理临时文件夹时出错: {str(e)}")
        print("\n")
    
    total_end_time = time.time()
    print(f"🎉 所有处理完成! 总用时: {total_end_time - total_start_time:.2f}秒")

if __name__ == "__main__":
    # 初始化CUDA上下文（如果可用）
    if cuda.is_available():
        # 获取当前设备
        device = cuda.get_current_device()
        # 获取或创建上下文
        ctx = cuda.current_context()
    
    # 处理阴影
    process_hourly_shadows(start_hour, end_hour, use_chunked=True)
    
    # 程序结束时清理CUDA资源 
    if cuda.is_available():
        ctx.reset()