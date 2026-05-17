import numpy as np
import pandas as pd
import os
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

file_path = r'C:\Users\39221\Desktop\数学建模大赛\正式比赛\2026_B题\数据\附件1.xlsx'
output_dir = r'C:\Users\39221\Desktop\数学建模大赛\正式比赛\2026_B题\图表'

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 数据读取函数
def load_data_excel(excel_path, sheet_name):
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    df.columns = ['time', 'x', 'y']
    return df['time'].values, df['x'].values, df['y'].values

# 2. 构造样条插值函数
def create_spline(t, x, y):
    """
    返回两个样条函数: spline_x(t), spline_y(t)
    """
    spline_x = CubicSpline(t, x, bc_type='natural')
    spline_y = CubicSpline(t, y, bc_type='natural')
    return spline_x, spline_y

# 3. 估计时间偏差 Δt
def estimate_dt(spline1_x, spline1_y, t1_min, t1_max,
                spline2_x, spline2_y, t2_min, t2_max):
    """
    通过最小化重叠区间内的轨迹差异来估计时间偏差
    """
    # 允许搜索的最大时间偏移范围：确保两条轨迹在时间轴上还有重叠可能
    search_min = t1_min - t2_max
    search_max = t1_max - t2_min

    # 重叠区间
    t_overlap_start = max(t1_min, t2_min)
    t_overlap_end = min(t1_max, t2_max)
    
    if t_overlap_start >= t_overlap_end:
        raise ValueError("两种数据没有时间重叠区域")
    
    print(f"重叠时间区间: [{t_overlap_start:.3f}, {t_overlap_end:.3f}] 秒")
    
    def error_function(dt, sample_step=0.01):
        """计算给定dt下的均方根误差"""
        
        # 在重叠区间内采样
        t_samples = np.arange(t_overlap_start, t_overlap_end, sample_step)
        
        # 方式1的位置
        x1 = spline1_x(t_samples)
        y1 = spline1_y(t_samples)
        
        # 方式2平移后的位置
        t2_shifted = t_samples + dt
        
        # 只保留在方式2有效范围内的点
        valid = (t2_shifted >= t2_min) & (t2_shifted <= t2_max)
        
        if np.sum(valid) < 10:
            return 1e10  # 无效时返回大值
        
        x2 = spline2_x(t2_shifted[valid])
        y2 = spline2_y(t2_shifted[valid])
        
        # 计算均方根误差（RMSE）
        rmse = np.sqrt(np.mean((x1[valid] - x2)**2 + (y1[valid] - y2)**2))
        return rmse
    
    # 先粗搜，再在最优附近精搜，避免多峰目标函数把优化器带到局部最优
    coarse_grid = np.linspace(search_min, search_max, 321)
    coarse_errors = np.array([error_function(dt, sample_step=0.1) for dt in coarse_grid])
    best_index = int(np.argmin(coarse_errors))
    coarse_best_dt = coarse_grid[best_index]

    local_left = max(search_min, coarse_best_dt - (search_max - search_min) / 321)
    local_right = min(search_max, coarse_best_dt + (search_max - search_min) / 321)

    result = minimize_scalar(
        lambda dt: error_function(dt, sample_step=0.01),
        bounds=(local_left, local_right),
        method='bounded',
        options={'xatol': 1e-8}
    )

    return result.x, error_function(result.x)

# 4. 生成10Hz融合轨迹
def generate_10hz_trajectory(spline1_x, spline1_y, t1_min, t1_max,
                             spline2_x, spline2_y, t2_min, t2_max,
                             dt, start_time, end_time):
    """
    生成10Hz的融合位置轨迹
    """
    # 生成0.1秒间隔的时间点
    t_10hz = np.arange(start_time, end_time, 0.1)
    
    x_fused = []
    y_fused = []
    
    for t in t_10hz:
        # 检查是否在各自的有效范围内
        valid1 = (t >= t1_min) and (t <= t1_max)
        valid2 = ((t + dt) >= t2_min) and ((t + dt) <= t2_max)
        
        if valid1 and valid2:
            # 两者都有：取平均
            x_fused.append((spline1_x(t) + spline2_x(t + dt)) / 2)
            y_fused.append((spline1_y(t) + spline2_y(t + dt)) / 2)
        elif valid1:
            # 只有方式1
            x_fused.append(spline1_x(t))
            y_fused.append(spline1_y(t))
        elif valid2:
            # 只有方式2
            x_fused.append(spline2_x(t + dt))
            y_fused.append(spline2_y(t + dt))
        else:
            # 超出范围：线性外推（或填充NaN）
            x_fused.append(np.nan)
            y_fused.append(np.nan)
    
    return t_10hz, np.array(x_fused), np.array(y_fused)

# 5. 主程序
def main():
    
    # 方式2：假设为Excel，两个sheet
    t1, x1, y1 = load_data_excel(file_path, '方式1(4Hz)')
    t2, x2, y2 = load_data_excel(file_path, '方式2(5Hz)')
    
    print("数据读取完成")
    print(f"(4Hz): {len(t1)} 个点, 时间范围 [{t1.min():.3f}, {t1.max():.3f}] 秒")
    print(f"(5Hz): {len(t2)} 个点, 时间范围 [{t2.min():.3f}, {t2.max():.3f}] 秒")

    # 构造样条插值
    spline1_x, spline1_y = create_spline(t1, x1, y1)
    spline2_x, spline2_y = create_spline(t2, x2, y2)
    
    # 获取时间范围
    t1_min, t1_max = t1.min(), t1.max()
    t2_min, t2_max = t2.min(), t2.max()
    
    # 估计时间偏差
    dt, rmse = estimate_dt(spline1_x, spline1_y, t1_min, t1_max,
                           spline2_x, spline2_y, t2_min, t2_max)
    
    print(f"\n{'='*50}")
    print(f"估计结果:")
    print(f"  时间偏差 Δt = {dt:.8f} 秒")
    print(f"  对齐后RMSE = {rmse:.8f} 米")
    print(f"  说明: 方式2的真实时间 = 记录时间 + {dt:.8f} 秒")
    print(f"{'='*50}\n")
    
    # 生成10Hz轨迹
    start_time = min(t1_min, t2_min)
    end_time = max(t1_max, t2_max)
    
    t_10hz, x_fused, y_fused = generate_10hz_trajectory(
        spline1_x, spline1_y, t1_min, t1_max,
        spline2_x, spline2_y, t2_min, t2_max,
        dt, start_time, end_time
    )
    
    # 保存结果
    result_df = pd.DataFrame({
        'time_s': t_10hz,
        'x_m': x_fused,
        'y_m': y_fused
    })
    result_df.to_csv('10hz_trajectory_Q1.csv', index=False)
    result_df.to_excel('10hz_trajectory_Q1.xlsx', index=False)
    
    print(f"已生成 {len(t_10hz)} 个10Hz轨迹点")
    print(f"保存至: 10hz_trajectory_Q1.csv 和 10hz_trajectory_Q1.xlsx")
    
    # 可选：可视化
    plt.figure(figsize=(12, 5))
    
    # 左图：对齐前
    plt.subplot(1, 2, 1)
    t_plot = np.arange(max(t1_min, t2_min), min(t1_max, t2_max), 0.01)
    plt.plot(spline1_x(t_plot), spline1_y(t_plot), 'b-', label='4Hz', alpha=0.7)
    plt.plot(spline2_x(t_plot), spline2_y(t_plot), 'r-', label='5Hz', alpha=0.7)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('对齐前')
    plt.legend()
    plt.axis('equal')
    
    # 右图：对齐后
    plt.subplot(1, 2, 2)
    plt.plot(spline1_x(t_plot), spline1_y(t_plot), 'b-', label='4Hz', alpha=0.7)
    plt.plot(spline2_x(t_plot + dt), spline2_y(t_plot + dt), 'r-', label='5Hz', alpha=0.7)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title(f'对齐后 (Δt = {dt:.6f} s)')
    plt.legend()
    plt.axis('equal')
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '问题一.png'), dpi=1000, bbox_inches='tight')
    plt.close()

    plt.plot(x_fused, y_fused, 'g.-', label='10Hz融合轨迹', alpha=0.7)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('10Hz融合轨迹')
    plt.legend()
    plt.axis('equal')

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, '问题一_10Hz融合轨迹.png'), dpi=1000, bbox_inches='tight')
    plt.close()
    print("可视化图片已保存: 问题一.png")

if __name__ == '__main__':
    main()