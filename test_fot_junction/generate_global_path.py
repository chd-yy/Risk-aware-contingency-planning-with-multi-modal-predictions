import numpy as np

x = [
    15.9923642,         6.06495,
    4.85655,            3.4265,            2.32685,
    1.10915,            -0.20755,            -1.58235,
    -2.96815,            -4.98995,            -6.89705,
    -8.10415,            -54.06815,            -100.0234
    ]

y = [
    -21.86624793,        -6.12275,
    -4.42595,            -2.91335,            -2.0663,
    -1.40115,            -0.96475,            -0.7815,
    -0.8549,            -1.3449,            -2.181,
    -2.8721,            -31.6067,            -60.3365
    ]


def densify_polyline(x, y, spacing=1.0):
    """
    按固定间距对折线进行加密。

    参数
    ----
    x, y : list 或 np.ndarray
        原始路径点
    spacing : float
        加密后的目标点间距（单位通常是米）

    返回
    ----
    x_dense, y_dense : list, list
        加密后的路径点
    """
    pts = np.column_stack((x, y)).astype(float)

    if len(pts) < 2:
        return list(x), list(y)

    # 相邻点距离
    seg_vecs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    # 去掉长度为 0 的重复段
    keep = np.hstack(([True], seg_lens > 1e-9))
    pts = pts[keep]

    if len(pts) < 2:
        return [pts[0, 0]], [pts[0, 1]]

    seg_vecs = pts[1:] - pts[:-1]
    seg_lens = np.linalg.norm(seg_vecs, axis=1)

    # 累计弧长
    s = np.zeros(len(pts))
    s[1:] = np.cumsum(seg_lens)
    total_len = s[-1]

    # 新的采样弧长
    s_dense = np.arange(0.0, total_len, spacing)
    if total_len - s_dense[-1] > 1e-9:
        s_dense = np.append(s_dense, total_len)

    # 分别对 x(s), y(s) 做一维线性插值
    x_dense = np.interp(s_dense, s, pts[:, 0])
    y_dense = np.interp(s_dense, s, pts[:, 1])

    return x_dense.tolist(), y_dense.tolist()


# 示例：按 2 米间距加密
x_dense, y_dense = densify_polyline(x, y, spacing=2.0)

print("x_dense = [")
for v in x_dense:
    print(f"    {v},")
print("]")

print("\ny_dense = [")
for v in y_dense:
    print(f"    {v},")
print("]")
