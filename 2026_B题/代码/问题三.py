from __future__ import annotations
from pathlib import Path
from types import ModuleType
import numpy as np
import pandas as pd
from filterpy.kalman import ExtendedKalmanFilter

def load_sheet(path: Path, sheet_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	df = pd.read_excel(path, sheet_name=sheet_name)
	if df.shape[1] < 3:
		raise ValueError("Expected at least 3 columns: time, x, y")
	t = df.iloc[:, 0].to_numpy(float)
	x = df.iloc[:, 1].to_numpy(float)
	y = df.iloc[:, 2].to_numpy(float)
	return t, x, y


def compute_mse(
	t_ref: np.ndarray,
	x_ref: np.ndarray,
	y_ref: np.ndarray,
	t_other: np.ndarray,
	x_other: np.ndarray,
	y_other: np.ndarray,
	shift: float,
) -> tuple[float, int]:
	t_shift = t_other + shift
	lo = max(t_ref[0], t_shift[0])
	hi = min(t_ref[-1], t_shift[-1])
	if hi <= lo:
		return float("inf"), 0
	mask = (t_ref >= lo) & (t_ref <= hi)
	if mask.sum() < 5:
		return float("inf"), int(mask.sum())
	x_interp = np.interp(t_ref[mask], t_shift, x_other)
	y_interp = np.interp(t_ref[mask], t_shift, y_other)
	mse = np.mean((x_ref[mask] - x_interp) ** 2 + (y_ref[mask] - y_interp) ** 2)
	return float(mse), int(mask.sum())


def estimate_time_shift(
	t_ref: np.ndarray,
	x_ref: np.ndarray,
	y_ref: np.ndarray,
	t_other: np.ndarray,
	x_other: np.ndarray,
	y_other: np.ndarray,
	coarse_step: float = 0.1,
	fine_step: float = 0.01,
) -> tuple[float, float, int]:
	min_shift = t_ref[0] - t_other[-1]
	max_shift = t_ref[-1] - t_other[0]
	if max_shift < min_shift:
		raise ValueError("No valid overlap after shifting")

	best_shift = None
	best_mse = float("inf")
	best_n = 0

	for s in np.arange(min_shift, max_shift + coarse_step, coarse_step):
		mse, n = compute_mse(t_ref, x_ref, y_ref, t_other, x_other, y_other, float(s))
		if mse < best_mse:
			best_shift, best_mse, best_n = float(s), mse, n

	if best_shift is None:
		raise ValueError("Failed to estimate time shift")

	fine_start = best_shift - 1.0
	fine_end = best_shift + 1.0
	for s in np.arange(fine_start, fine_end + fine_step, fine_step):
		mse, n = compute_mse(t_ref, x_ref, y_ref, t_other, x_other, y_other, float(s))
		if mse < best_mse:
			best_shift, best_mse, best_n = float(s), mse, n

	return best_shift, best_mse, best_n


def estimate_bias(
	t_ref: np.ndarray,
	x_ref: np.ndarray,
	y_ref: np.ndarray,
	t_other: np.ndarray,
	x_other: np.ndarray,
	y_other: np.ndarray,
	shift: float,
) -> tuple[float, float, float, float]:
	t_shift = t_other + shift
	lo = max(t_ref[0], t_shift[0])
	hi = min(t_ref[-1], t_shift[-1])
	if hi <= lo:
		raise ValueError("No overlap after applying time shift")
	mask = (t_ref >= lo) & (t_ref <= hi)
	if mask.sum() < 5:
		raise ValueError("Not enough overlapping samples for bias estimation")

	x_interp = np.interp(t_ref[mask], t_shift, x_other)
	y_interp = np.interp(t_ref[mask], t_shift, y_other)
	dx = x_ref[mask] - x_interp
	dy = y_ref[mask] - y_interp

	bias_x = float(np.mean(dx))
	bias_y = float(np.mean(dy))
	rms_before = float(np.sqrt(np.mean(dx**2 + dy**2)))
	rms_after = float(np.sqrt(np.mean((dx - bias_x) ** 2 + (dy - bias_y) ** 2)))
	return bias_x, bias_y, rms_before, rms_after


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
	if window <= 1 or len(values) < window:
		return values.copy()
	kernel = np.ones(window, dtype=float) / float(window)
	return np.convolve(values, kernel, mode="same")


def estimate_measurement_noise(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
	if len(x) < 3:
		return 1e-4, 1e-4
	window = 9 if len(x) >= 9 else (5 if len(x) >= 5 else 3)
	x_s = moving_average(x, window)
	y_s = moving_average(y, window)
	res_x = x - x_s
	res_y = y - y_s
	r_x = float(np.var(res_x))
	r_y = float(np.var(res_y))
	r_floor = 1e-4
	return max(r_x, r_floor), max(r_y, r_floor)


def filter_positions_ekf(
	t: np.ndarray,
	x: np.ndarray,
	y: np.ndarray,
	q_base: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	order = np.argsort(t)
	t_sorted = t[order]
	x_sorted = x[order]
	y_sorted = y[order]

	r_x, r_y = estimate_measurement_noise(x_sorted, y_sorted)
	ekf = ExtendedKalmanFilter(dim_x=4, dim_z=2)
	ekf.x = np.array([x_sorted[0], y_sorted[0], 0.0, 0.0], dtype=float)
	ekf.P = np.eye(4) * 100.0
	ekf.R = np.array([[r_x, 0.0], [0.0, r_y]], dtype=float)
	ekf.Q = np.eye(4) * q_base

	def hx(state: np.ndarray) -> np.ndarray:
		return np.array([state[0], state[1]], dtype=float)

	def hjacobian(_state: np.ndarray) -> np.ndarray:
		return np.array(
			[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
			dtype=float,
		)

	last_time = t_sorted[0]
	x_filt = np.zeros_like(x_sorted)
	y_filt = np.zeros_like(y_sorted)

	for i in range(len(t_sorted)):
		dt = t_sorted[i] - last_time
		if dt <= 0:
			dt = 1e-6
		last_time = t_sorted[i]
		ekf.F = np.array(
			[
				[1.0, 0.0, dt, 0.0],
				[0.0, 1.0, 0.0, dt],
				[0.0, 0.0, 1.0, 0.0],
				[0.0, 0.0, 0.0, 1.0],
			],
			dtype=float,
		)
		ekf.Q = np.eye(4, dtype=float) * q_base * dt
		ekf.predict()
		z = np.array([x_sorted[i], y_sorted[i]], dtype=float)
		ekf.update(z, HJacobian=hjacobian, Hx=hx)
		x_filt[i] = ekf.x[0]
		y_filt[i] = ekf.x[1]

	return t_sorted, x_filt, y_filt


def interp_with_nan(t_target: np.ndarray, t_src: np.ndarray, v_src: np.ndarray) -> np.ndarray:
	values = np.interp(t_target, t_src, v_src).astype(float)
	mask = (t_target < t_src[0]) | (t_target > t_src[-1])
	values[mask] = np.nan
	return values


def fuse_to_10hz(
	t_a: np.ndarray,
	x_a: np.ndarray,
	y_a: np.ndarray,
	t_b: np.ndarray,
	x_b: np.ndarray,
	y_b: np.ndarray,
	shift: float,
	bias_x: float,
	bias_y: float,
	step: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
	t_b_shift = t_b + shift
	start = max(t_a[0], t_b_shift[0])
	end = min(t_a[-1], t_b_shift[-1])
	if end <= start:
		raise ValueError("No overlap for fusion")

	t_grid = np.arange(start, end + step / 2, step)

	x_a_i = interp_with_nan(t_grid, t_a, x_a)
	y_a_i = interp_with_nan(t_grid, t_a, y_a)
	x_b_i = interp_with_nan(t_grid, t_b_shift, x_b) + bias_x
	y_b_i = interp_with_nan(t_grid, t_b_shift, y_b) + bias_y

	x_fused = np.nanmean(np.vstack([x_a_i, x_b_i]), axis=0)
	y_fused = np.nanmean(np.vstack([y_a_i, y_b_i]), axis=0)
	return t_grid, x_a_i, y_a_i, x_b_i, y_b_i, x_fused, y_fused


def setup_matplotlib() -> ModuleType:
	import matplotlib.pyplot as plt

	plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
	plt.rcParams["axes.unicode_minus"] = False
	return plt


def plot_before(
	plt: ModuleType,
	t: np.ndarray,
	x_a: np.ndarray,
	y_a: np.ndarray,
	x_b: np.ndarray,
	y_b: np.ndarray,
	output_path: Path,
) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
	axes[0].plot(t, x_a, label="A 传感器(滤波)")
	axes[0].plot(t, x_b, label="B 传感器(滤波+对齐)", alpha=0.85)
	axes[0].set_ylabel("X (m)")
	axes[0].set_title("融合前对比")
	axes[0].legend()

	axes[1].plot(t, y_a, label="A 传感器(滤波)")
	axes[1].plot(t, y_b, label="B 传感器(滤波+对齐)", alpha=0.85)
	axes[1].set_xlabel("时间 (s)")
	axes[1].set_ylabel("Y (m)")
	axes[1].legend()

	fig.tight_layout()
	fig.savefig(output_path, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def plot_after(
	plt: ModuleType,
	t: np.ndarray,
	x_a: np.ndarray,
	y_a: np.ndarray,
	x_b: np.ndarray,
	y_b: np.ndarray,
	x_fused: np.ndarray,
	y_fused: np.ndarray,
	output_path: Path,
) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
	axes[0].plot(t, x_a, label="A 传感器(滤波)")
	axes[0].plot(t, x_b, label="B 传感器(滤波+对齐)", alpha=0.8)
	axes[0].plot(t, x_fused, label="融合10Hz", linewidth=2.0)
	axes[0].set_ylabel("X (m)")
	axes[0].set_title("融合后对比")
	axes[0].legend()

	axes[1].plot(t, y_a, label="A 传感器(滤波)")
	axes[1].plot(t, y_b, label="B 传感器(滤波+对齐)", alpha=0.8)
	axes[1].plot(t, y_fused, label="融合10Hz", linewidth=2.0)
	axes[1].set_xlabel("时间 (s)")
	axes[1].set_ylabel("Y (m)")
	axes[1].legend()

	fig.tight_layout()
	fig.savefig(output_path, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def plot_fused_only(
	plt: ModuleType,
	t: np.ndarray,
	x_fused: np.ndarray,
	y_fused: np.ndarray,
	output_path: Path,
) -> None:
	fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
	axes[0].plot(t, x_fused, label="融合X", linewidth=1.8)
	axes[0].set_ylabel("X (m)")
	axes[0].set_title("融合后坐标")
	axes[0].legend()

	axes[1].plot(t, y_fused, label="融合Y", linewidth=1.8)
	axes[1].set_xlabel("时间 (s)")
	axes[1].set_ylabel("Y (m)")
	axes[1].legend()

	fig.tight_layout()
	fig.savefig(output_path, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def plot_xy_before(
	plt: ModuleType,
	x_a: np.ndarray,
	y_a: np.ndarray,
	x_b: np.ndarray,
	y_b: np.ndarray,
	output_path: Path,
) -> None:
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.plot(x_a, y_a, label="A 传感器(滤波)")
	ax.plot(x_b, y_b, label="B 传感器(滤波+对齐)", alpha=0.85)
	ax.set_title("融合前XY轨迹对比")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.axis("equal")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def plot_xy_raw(
	plt: ModuleType,
	x_a: np.ndarray,
	y_a: np.ndarray,
	x_b: np.ndarray,
	y_b: np.ndarray,
	output_path: Path,
) -> None:
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.plot(x_a, y_a, label="A 传感器(原始)")
	ax.plot(x_b, y_b, label="B 传感器(原始)", alpha=0.85)
	ax.set_title("原始XY轨迹对比")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.axis("equal")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def plot_xy_after(
	plt: ModuleType,
	x_a: np.ndarray,
	y_a: np.ndarray,
	x_b: np.ndarray,
	y_b: np.ndarray,
	x_fused: np.ndarray,
	y_fused: np.ndarray,
	output_path: Path,
) -> None:
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.plot(x_a, y_a, label="A 传感器(滤波)")
	ax.plot(x_b, y_b, label="B 传感器(滤波+对齐)", alpha=0.8)
	ax.plot(x_fused, y_fused, label="融合10Hz", linewidth=2.0)
	ax.set_title("融合后XY轨迹对比")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.axis("equal")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def plot_fused_xy_only(
	plt: ModuleType,
	x_fused: np.ndarray,
	y_fused: np.ndarray,
	output_path: Path,
) -> None:
	fig, ax = plt.subplots(figsize=(8, 8))
	ax.plot(x_fused, y_fused, label="融合10Hz", linewidth=2.0)
	ax.set_title("融合后XY坐标")
	ax.set_xlabel("X (m)")
	ax.set_ylabel("Y (m)")
	ax.axis("equal")
	ax.legend()
	fig.tight_layout()
	fig.savefig(output_path, dpi=1000, bbox_inches="tight")
	plt.close(fig)


def main() -> None:
	script_dir = Path(__file__).resolve().parent
	data_path = script_dir.parent / "数据" / "附件3.xlsx"
	if not data_path.exists():
		raise FileNotFoundError(f"Data file not found: {data_path}")

	xls = pd.ExcelFile(data_path)
	if len(xls.sheet_names) < 2:
		raise ValueError("Expected at least two sheets for sensor A and B")

	sheet_a = xls.sheet_names[0]
	sheet_b = xls.sheet_names[1]

	t_a_raw, x_a_raw, y_a_raw = load_sheet(data_path, sheet_a)
	t_b_raw, x_b_raw, y_b_raw = load_sheet(data_path, sheet_b)

	t_a, x_a, y_a = filter_positions_ekf(t_a_raw, x_a_raw, y_a_raw)
	t_b, x_b, y_b = filter_positions_ekf(t_b_raw, x_b_raw, y_b_raw)

	shift, mse, n = estimate_time_shift(t_a, x_a, y_a, t_b, x_b, y_b)
	bias_x, bias_y, rms_before, rms_after = estimate_bias(t_a, x_a, y_a, t_b, x_b, y_b, shift)

	t_grid, x_a_i, y_a_i, x_b_i, y_b_i, x_fused, y_fused = fuse_to_10hz(
		t_a,
		x_a,
		y_a,
		t_b,
		x_b,
		y_b,
		shift,
		bias_x,
		bias_y,
	)

	output_dir = script_dir.parent / "图表"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_file = script_dir.parent / "结果输出"
	output_file.mkdir(parents=True, exist_ok=True)

	output_data = pd.DataFrame(
		{
			"时间(s)": t_grid,
			"A_X滤波(m)": x_a_i,
			"A_Y滤波(m)": y_a_i,
			"B_X滤波对齐(m)": x_b_i,
			"B_Y滤波对齐(m)": y_b_i,
			"融合_X(m)": x_fused,
			"融合_Y(m)": y_fused,
		}
	)
	output_excel = output_file / "融合10Hz数据.xlsx"
	output_data.to_excel(output_excel, index=False)

	plt = setup_matplotlib()
	plot_xy_raw(plt, x_a_raw, y_a_raw, x_b_raw, y_b_raw, output_dir / "问题三_原始XY轨迹对比.png")
	plot_before(plt, t_grid, x_a_i, y_a_i, x_b_i, y_b_i, output_dir / "问题三_融合前对比.png")
	plot_after(
		plt,
		t_grid,
		x_a_i,
		y_a_i,
		x_b_i,
		y_b_i,
		x_fused,
		y_fused,
		output_dir / "问题三_融合后对比.png",
	)
	plot_fused_only(plt, t_grid, x_fused, y_fused, output_dir / "问题三_融合后坐标.png")
	plot_xy_before(plt, x_a_i, y_a_i, x_b_i, y_b_i, output_dir / "问题三_融合前XY轨迹对比.png")
	plot_xy_after(
		plt,
		x_a_i,
		y_a_i,
		x_b_i,
		y_b_i,
		x_fused,
		y_fused,
		output_dir / "问题三_融合后XY轨迹对比.png",
	)
	plot_fused_xy_only(plt, x_fused, y_fused, output_dir / "问题三_融合后XY坐标.png")

	print("参考传感器(A)工作表:", sheet_a)
	print("待对齐传感器(B)工作表:", sheet_b)
	print("已使用扩展卡尔曼滤波对位置数据进行去噪")
	print("估计时间偏差(B -> A):", shift)
	print("最佳时间偏差下的均方误差:", mse)
	print("参与对齐的重叠样本数:", n)
	print("估计系统误差(平移 dx, dy):", bias_x, bias_y)
	print("偏移修正前RMS:", rms_before)
	print("偏移修正后RMS:", rms_after)
	print("融合10Hz数据已输出:", output_excel)
	print("融合前对比图已输出:", output_dir / "问题三_融合前对比.png")
	print("融合后对比图已输出:", output_dir / "问题三_融合后对比.png")
	print("融合后坐标图已输出:", output_dir / "问题三_融合后坐标.png")
	print("原始XY轨迹对比图已输出:", output_dir / "问题三_原始XY轨迹对比.png")
	print("融合前XY轨迹对比图已输出:", output_dir / "问题三_融合前XY轨迹对比.png")
	print("融合后XY轨迹对比图已输出:", output_dir / "问题三_融合后XY轨迹对比.png")
	print("融合后XY坐标图已输出:", output_dir / "问题三_融合后XY坐标.png")

if __name__ == "__main__":
	main()