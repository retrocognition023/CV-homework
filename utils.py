# -*- coding: utf-8 -*-
# 给 homework2.py 用的几个小函数
# 只用 numpy 和 scipy，角点、匹配这些没调现成接口
import numpy as np
from scipy import ndimage


def to_gray(a):
    """彩色转灰度，顺便转成 float，uint8 相减会溢出。"""
    a = np.asarray(a)
    if a.ndim == 3:
        b = a[..., 0].astype(np.float64)
        g = a[..., 1].astype(np.float64)
        r = a[..., 2].astype(np.float64)
        return 0.114 * b + 0.587 * g + 0.299 * r
    return a.astype(np.float64)


def win1(n, s):
    """长度 n 的一维高斯窗，和归一到 1。"""
    t = np.arange(n) - (n - 1) / 2.0
    w = np.exp(-t * t / (2.0 * s * s))
    return w / w.sum()


def smooth2(a, n, s):
    """先竖着卷一次，再横着卷一次，边界 reflect。"""
    w = win1(n, s)
    b = ndimage.convolve1d(a, w, axis=0, mode='reflect')
    return ndimage.convolve1d(b, w, axis=1, mode='reflect')


def norm_pts(p):
    """点集重心移到原点、平均距离缩到 sqrt(2)。

    返回归一化坐标和变换矩阵 t，后面靠 t 变回去。
    不归一化的话，几百量级的像素坐标容易让 DLT 矩阵病态。
    """
    p = np.asarray(p, dtype=np.float64)
    c = p.mean(axis=0)
    q = p - c
    d = np.sqrt((q * q).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    t = np.array([[s, 0.0, -s * c[0]],
                  [0.0, s, -s * c[1]],
                  [0.0, 0.0, 1.0]])
    q = np.column_stack([p, np.ones(len(p))])
    return (t @ q.T).T[:, :2], t


def reproj_err(h, p1, p2):
    """把 p1 用 h 投过去，和 p2 比，逐点算距离。"""
    q = np.column_stack([np.asarray(p1, np.float64), np.ones(len(p1))])
    v = (h @ q.T).T
    z = v[:, 2]
    out = np.full(len(v), np.inf)
    ok = np.abs(z) > 1e-12
    out[ok] = np.sqrt(((v[ok, :2] / z[ok, None] -
                       np.asarray(p2, np.float64)[ok]) ** 2).sum(axis=1))
    return out


def ransac_with_inliers(p1, p2, fit, n_iter=1000, tol=5.0, seed=0):
    """RANSAC 求单应，把最好模型的内点掩码也留下来给报告画图。

    fit 由调用方传进来，utils.py 不用反过来 import 主程序。
    返回 H、内点布尔数组、内点个数。
    """
    p1 = np.asarray(p1, np.float64)
    p2 = np.asarray(p2, np.float64)
    n = len(p1)
    rng = np.random.default_rng(seed)
    best_h, best_m, best_c = None, None, -1

    for _ in range(n_iter):
        idx = rng.choice(n, 4, replace=False)
        try:
            h = fit(p1[idx].tolist(), p2[idx].tolist())
        except Exception:
            continue
        if h is None or not np.all(np.isfinite(h)):
            continue
        e = reproj_err(h, p1, p2)
        m = e < tol
        c = int(m.sum())
        if c > best_c:
            best_h, best_m, best_c = h, m, c

    if best_m is None or best_c < 4:
        h = fit(p1.tolist(), p2.tolist())
        m = np.ones(n, dtype=bool)
        return h, m, n

    # 用全部内点再拟合一次，比 4 个点稳
    h = fit(p1[best_m].tolist(), p2[best_m].tolist())
    m = reproj_err(h, p1, p2) < tol
    return h, m, int(m.sum())
