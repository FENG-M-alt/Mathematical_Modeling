import os
import pandas as pd
import numpy as np
from filterpy.kalman import KalmanFilter
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

file_path = r"C:\Users\39221\Desktop\数学建模大赛\正式比赛\2026_B题\数据\附件2.xlsx"
out_path = r"C:\Users\39221\Desktop\数学建模大赛\正式比赛\2026_B题\结果输出"

os.makedirs(out_path, exist_ok=True)

out_lable_path = r"C:\Users\39221\Desktop\数学建模大赛\正式比赛\2026_B题\图表"
os.makedirs(out_lable_path, exist_ok=True)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

sensor1 = {
    "sheet": "方式1(4Hz)",
    "time_col": "时间(s)",
    "x_col": "X坐标(m)",
    "y_col": "Y坐标(m)",
    "label_cn": "传感器1（4Hz）"
}

sensor2 = {
    "sheet": "方式2(5Hz)",
    "time_col": "时间(s)",
    "x_col": "X坐标(m)",
    "y_col": "Y坐标(m)",
    "label_cn": "传感器2（5Hz）"
}

# 过程噪声基础值（可调）
Q_BASE = np.eye(4) * 0.1

# 最大允许时间偏移（秒），将搜索区间裁剪到 [-MAX_SHIFT, MAX_SHIFT]
# 若不希望限制，可设置为 None
MAX_SHIFT = 100.0

# 自适应异常门限（残差超过 3σ 视为异常）
SIGMA_GATE = 3.0

# 异常时 R 放大倍数
FAULTY_FACTOR = 30.0

# 滤波函数：处理单个传感器数据并返回滤波结果
def filter_single_sensor(df, time_col, x_col, y_col, label):
    """返回滤波后的 DataFrame 和估计矩阵"""
    # 重命名列以便统一处理
    df = df.rename(columns={time_col: 'time', x_col: 'x', y_col: 'y'})
    df = df.dropna(subset=['x', 'y']).reset_index(drop=True)

    time = df['time'].values
    x_raw = df['x'].values
    y_raw = df['y'].values

    # ---- 自动估计观测噪声 R ----
    def estimate_R_from_residual(x, y, window=11, polyorder=2):
        smooth_x = savgol_filter(x, window, polyorder)
        smooth_y = savgol_filter(y, window, polyorder)
        res_x = x - smooth_x
        res_y = y - smooth_y
        return np.cov(res_x, res_y)

    def estimate_R_from_static(x, y, threshold=0.01):
        dx = np.diff(x)
        dy = np.diff(y)
        mask = (np.abs(dx) < threshold) & (np.abs(dy) < threshold)
        if np.sum(mask) < 20:
            return None
        return np.cov(x[:-1][mask], y[:-1][mask])

    R_est = estimate_R_from_static(x_raw, y_raw)
    if R_est is None:
        R_est = estimate_R_from_residual(x_raw, y_raw)
    R = np.diag(np.diag(R_est))
    R = np.maximum(R, np.eye(2) * 0.01)

    # ---- 初始化卡尔曼滤波器 ----
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])
    kf.x = np.array([x_raw[0], y_raw[0], 0., 0.])
    kf.P = np.eye(4) * 500
    last_time = time[0]

    # ---- 自适应更新 ----
    def smart_update(kf, z, R_nom):
        y = z - kf.H @ kf.x
        S = kf.H @ kf.P @ kf.H.T + R_nom
        std = np.sqrt(np.diag(S))
        if np.any(np.abs(y) > SIGMA_GATE * std):
            kf.update(z, R=R * FAULTY_FACTOR)
            return True
        else:
            kf.update(z, R=R_nom)
            return False

    # ---- 主循环 ----
    N = len(df)
    estimates = np.zeros((N, 4))
    anomalies = np.zeros(N, dtype=bool)

    for i in range(N):
        dt = time[i] - last_time
        if dt <= 0:
            dt = 1e-6
        last_time = time[i]

        kf.F = np.array([[1, 0, dt, 0],
                         [0, 1, 0, dt],
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]])
        kf.Q = Q_BASE * dt
        kf.predict()

        z = np.array([x_raw[i], y_raw[i]])
        anomalies[i] = smart_update(kf, z, R)
        estimates[i] = kf.x

    # ---- 组装输出 ----
    df_out = df.copy()
    df_out['kf_x'] = estimates[:, 0]
    df_out['kf_y'] = estimates[:, 1]
    df_out['kf_vx'] = estimates[:, 2]
    df_out['kf_vy'] = estimates[:, 3]
    df_out['anomaly'] = anomalies
    return df_out

def create_spline(t, x, y):
    """
    返回两个样条函数: spline_x(t), spline_y(t)
    """
    spline_x = CubicSpline(t, x, bc_type='natural')
    spline_y = CubicSpline(t, y, bc_type='natural')
    return spline_x, spline_y

def estimate_dt(spline1_x, spline1_y, t1_min, t1_max,
                spline2_x, spline2_y, t2_min, t2_max):
    """
    通过最小化重叠区间内的轨迹差异来估计时间偏差
    """
    # 允许搜索的最大时间偏移范围（初始）：确保两条轨迹在时间轴上还有重叠可能
    search_min = t1_min - t2_max
    search_max = t1_max - t2_min

    # 如果设置了全局的 MAX_SHIFT，则将搜索区间裁剪到 [-MAX_SHIFT, MAX_SHIFT]
    try:
        max_shift = float(MAX_SHIFT) if MAX_SHIFT is not None else None
    except Exception:
        max_shift = None
    if max_shift is not None:
        # 将 search_min/search_max 裁剪到允许的最大偏移范围
        search_min = max(search_min, -max_shift)
        search_max = min(search_max, max_shift)

    # 重叠区间
    t_overlap_start = max(t1_min, t2_min)
    t_overlap_end = min(t1_max, t2_max)
    
    if t_overlap_start >= t_overlap_end:
        raise ValueError("两种数据没有时间重叠区域")

    print(f"重叠时间区间: [{t_overlap_start:.3f}, {t_overlap_end:.3f}] 秒")
    # 打印并提示搜索区间
    print(f"搜索初始区间: [{search_min:.6f}, {search_max:.6f}] 秒（已应用 MAX_SHIFT={MAX_SHIFT}）")
    if search_min > search_max:
        raise ValueError("在应用 MAX_SHIFT 限制后，搜索区间为空，请增大 MAX_SHIFT 或检查数据时间范围。")
    
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

def compute_systematic_error(df1_filtered, df2_filtered, dt,
                            sensor1_name, sensor2_name,
                            out_path, n_samples=1000):
        """
        计算两个副表（已滤波）在给定时间偏移 dt 下的固定系统误差。

        输入:
            df1_filtered, df2_filtered: 包含列 ['time','kf_x','kf_y'] 的 DataFrame
            dt: 方式2 相对于方式1 的时间偏移（方式1时间 = 方式2时间 + dt）
            sensor1_name, sensor2_name: 中文名称，用于输出文件名和打印
            out_path: 保存结果的目录
            n_samples: 采样点数

        输出: 返回 (summary_df, detail_df)
        并将统计结果与逐点差异保存为 Excel 文件。
        """
        # 创建样条
        spline1_x, spline1_y = create_spline(df1_filtered['time'], df1_filtered['kf_x'], df1_filtered['kf_y'])
        spline2_x, spline2_y = create_spline(df2_filtered['time'], df2_filtered['kf_x'], df2_filtered['kf_y'])

        t1_min, t1_max = df1_filtered['time'].min(), df1_filtered['time'].max()
        t2_min, t2_max = df2_filtered['time'].min(), df2_filtered['time'].max()

        # 可比较的时间区间
        t_sys_start = max(t1_min, t2_min - dt)
        t_sys_end = min(t1_max, t2_max - dt)
        if t_sys_start >= t_sys_end:
                raise ValueError("没有足够的重叠时间用于计算系统误差。")

        t_common = np.linspace(t_sys_start, t_sys_end, n_samples)
        x1c = spline1_x(t_common)
        y1c = spline1_y(t_common)
        x2c = spline2_x(t_common + dt)
        y2c = spline2_y(t_common + dt)

        dx = x1c - x2c
        dy = y1c - y2c

        mean_dx, mean_dy = np.mean(dx), np.mean(dy)
        med_dx, med_dy = np.median(dx), np.median(dy)
        std_dx, std_dy = np.std(dx), np.std(dy)
        rms_xy = np.sqrt(np.mean(dx**2 + dy**2))

        summary_df = pd.DataFrame({
                '指标': ['平均偏差ΔX(m)', '平均偏差ΔY(m)', '中位数ΔX(m)', '中位数ΔY(m)',
                             '标准差ΔX(m)', '标准差ΔY(m)', '逐点RMS(m)'],
                '值': [mean_dx, mean_dy, med_dx, med_dy, std_dx, std_dy, rms_xy]
        })

        detail_df = pd.DataFrame({'时间(s)': t_common, 'ΔX(m)': dx, 'ΔY(m)': dy})

        # 保存
        summary_file = os.path.join(out_path, f"系统误差统计_{sensor1_name}_vs_{sensor2_name}.xlsx")
        detail_file = os.path.join(out_path, f"系统误差逐点_{sensor1_name}_vs_{sensor2_name}.xlsx")
        summary_df.to_excel(summary_file, index=False)
        detail_df.to_excel(detail_file, index=False)

        print(f"系统误差统计已保存至 {summary_file}")
        print(f"逐点差异已保存至 {detail_file}")
        # 返回统计表、逐点差异表及平均偏差（ΔX, ΔY），供后续融合使用
        return summary_df, detail_df, (mean_dx, mean_dy)

def fuse_data(df1_filtered, df2_filtered, dt, mean_bias, out_path, out_lable_path, freq=10.0):
    """
    根据时间偏差 dt 与系统误差 mean_bias，按 freq 频率融合数据并出图
    """
    s1x, s1y = create_spline(df1_filtered['time'], df1_filtered['kf_x'], df1_filtered['kf_y'])
    s2x, s2y = create_spline(df2_filtered['time'], df2_filtered['kf_x'], df2_filtered['kf_y'])

    t1_min, t1_max = df1_filtered['time'].min(), df1_filtered['time'].max()
    t2_min, t2_max = df2_filtered['time'].min(), df2_filtered['time'].max()

    # 可靠重叠区间
    t_start = max(t1_min, t2_min - dt)
    t_end = min(t1_max, t2_max - dt)

    if t_start >= t_end:
        raise ValueError("没有足够的重叠时间进行融合")

    dt_sample = 1.0 / freq
    t_common = np.arange(t_start, t_end, dt_sample)

    x1 = s1x(t_common)
    y1 = s1y(t_common)
    x2 = s2x(t_common + dt)
    y2 = s2y(t_common + dt)

    mean_dx, mean_dy = mean_bias
    x2_aligned = x2 + mean_dx
    y2_aligned = y2 + mean_dy

    fused_x = 0.5 * (x1 + x2_aligned)
    fused_y = 0.5 * (y1 + y2_aligned)

    vx = np.gradient(fused_x, dt_sample)
    vy = np.gradient(fused_y, dt_sample)

    fused_df = pd.DataFrame({
        '时间(s)': t_common,
        '融合X(m)': fused_x,
        '融合Y(m)': fused_y,
        '速度X(m/s)': vx,
        '速度Y(m/s)': vy,
        '传感器1验证X(m)': x1,
        '传感器1验证Y(m)': y1,
        '传感器2验证X_对齐(m)': x2_aligned,
        '传感器2验证Y_对齐(m)': y2_aligned
    })

    fused_file = os.path.join(out_path, "问题二_融合结果_10Hz.xlsx")
    fused_df.to_excel(fused_file, index=False)

    # 绘图
    plt.figure(figsize=(8, 8))
    plt.plot(x1, y1, 'g--', alpha=0.5, label='传感器1轨迹')
    plt.plot(x2_aligned, y2_aligned, 'c--', alpha=0.5, label='传感器2对齐轨迹')
    plt.plot(fused_x, fused_y, 'r-', lw=2, label='融合轨迹(10Hz)')
    plt.xlabel('X坐标(m)')
    plt.ylabel('Y坐标(m)')
    plt.title('两传感器融合轨迹及对比 (10Hz)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    plot_file = os.path.join(out_lable_path, "问题二_融合轨迹_10Hz_1.png")
    plt.savefig(plot_file, dpi=1000, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(fused_x, fused_y, 'r-', lw=2, label='融合轨迹(10Hz)')
    plt.xlabel('X坐标(m)')
    plt.ylabel('Y坐标(m)')
    plt.title('两传感器融合轨迹 (10Hz)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    plot_file = os.path.join(out_lable_path, "问题二_融合轨迹_10Hz_2.png")
    plt.savefig(plot_file, dpi=1000, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(8, 8))
    plt.plot(x1, y1, 'g--', alpha=0.5, label='传感器1轨迹')
    plt.plot(x2_aligned, y2_aligned, 'c--', alpha=0.5, label='传感器2对齐轨迹')
    plt.xlabel('X坐标(m)')
    plt.ylabel('Y坐标(m)')
    plt.title('两传感器轨迹及对比 (10Hz)')
    plt.axis('equal')
    plt.legend()
    plt.grid(True)

    plot_file = os.path.join(out_lable_path, "问题二_融合轨迹_10Hz_3.png")
    plt.savefig(plot_file, dpi=1000, bbox_inches='tight')
    plt.close()


    print(f"融合结果已保存至: {fused_file}")
    print(f"融合图表已保存至: {plot_file}")
    return fused_df

def filtering(sensor1, df1):
    for sensor, df_raw in [(sensor1, df1)]:
        print(f"正在处理 {sensor['label_cn']} ...")
        df_filtered = filter_single_sensor(
            df_raw,
            sensor['time_col'],
            sensor['x_col'],
            sensor['y_col'],
            sensor['label_cn']
        )

        # 将列名重命名为中文后保存（仅中文表头）
        df_save = df_filtered.rename(columns={
            'time': '时间(s)',
            'x': 'X坐标(m)',
            'y': 'Y坐标(m)',
            'kf_x': '滤波后X(m)',
            'kf_y': '滤波后Y(m)',
            'kf_vx': '速度X(m/s)',
            'kf_vy': '速度Y(m/s)',
            'anomaly': '异常'
        })

        # 保存到新 Excel 文件
        out_file = os.path.join(out_path, f"问题二_滤波结果_{sensor['label_cn']}.xlsx")
        df_save.to_excel(out_file, index=False)

        # 绘制轨迹图
        plt.figure(figsize=(8, 8))
        plt.plot(df_filtered['x'], df_filtered['y'], 'r.', alpha=0.3, label='原始')
        plt.plot(df_filtered['kf_x'], df_filtered['kf_y'], 'b-', lw=2, label='滤波后')
        plt.scatter(
            df_filtered.loc[df_filtered['anomaly'], 'kf_x'],
            df_filtered.loc[df_filtered['anomaly'], 'kf_y'],
            c='orange', marker='x', s=60, label='异常'
        )
        plt.xlabel('X坐标(m)'); plt.ylabel('Y坐标(m)')
        plt.title(f'单传感器滤波 - {sensor["label_cn"]}')
        plt.axis('equal')
        plt.legend()
        plt.grid(True)

        # 保存图表到指定目录
        plot_file = os.path.join(out_lable_path, f"问题二_图表_{sensor['label_cn']}.png")
        plt.savefig(plot_file, dpi=1000, bbox_inches='tight')
        print(f"图表已保存至 {plot_file}")
        plt.close()

        return df_filtered

def main():   
    sheets = [sensor1['sheet'], sensor2['sheet']]
    dfs = pd.read_excel(file_path, sheet_name=sheets)

    df1 = dfs[sensor1['sheet']]
    df2 = dfs[sensor2['sheet']]
    
    # 滤波处理
    df1_filtered = filtering(sensor1, df1)
    df2_filtered = filtering(sensor2, df2)

    # 创建样条函数
    spline1_x, spline1_y = create_spline(df1_filtered['time'], df1_filtered['kf_x'], df1_filtered['kf_y'])
    spline2_x, spline2_y = create_spline(df2_filtered['time'], df2_filtered['kf_x'], df2_filtered['kf_y'])

    t1_min, t1_max = df1_filtered['time'].min(), df1_filtered['time'].max()
    t2_min, t2_max = df2_filtered['time'].min(), df2_filtered['time'].max()

    # 估计时间偏差
    dt, rmse = estimate_dt(
        spline1_x, spline1_y,
        t1_min, t1_max,
        spline2_x, spline2_y,
        t2_min, t2_max
    )

    print(f"估计结果:")
    print(f"  时间偏差 Δt = {dt:.8f} 秒")
    print(f"  对齐后RMSE = {rmse:.8f} 米")

    summary_df, detail_df, mean_bias = compute_systematic_error(
        df1_filtered, df2_filtered, dt,
        sensor1['label_cn'], sensor2['label_cn'],
        out_path
    )
    print(f"平均偏差: ΔX = {mean_bias[0]:.6f} m, ΔY = {mean_bias[1]:.6f} m")

    # 融合数据到 10Hz 并生成图表
    fuse_data(
        df1_filtered, df2_filtered, dt, mean_bias,
        out_path, out_lable_path, freq=10.0
    )

if __name__ == "__main__":
    main()