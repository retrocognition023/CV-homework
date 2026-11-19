# -*- coding: utf-8 -*-
# 只做 Q6 bonus 2 的对比实验，不参与主流程，也没动 homework2.py 里的拼接函数
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import homework2 as hw
import utils

D = 'Problem2Images/'
os.makedirs('outputs', exist_ok=True)


def rgb(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def lap(a, lv=4):
    """从细到粗拆成 lv 层，最后一层留着最糊的那张。"""
    g = [a.astype(np.float32)]
    for _ in range(lv - 1):
        g.append(cv2.pyrDown(g[-1]))
    d = []
    for i in range(lv - 1):
        u = cv2.pyrUp(g[i + 1], dstsize=(g[i].shape[1], g[i].shape[0]))
        d.append(g[i] - u)
    d.append(g[-1])
    return d


def back(d):
    """从最粗那层往回加细节。"""
    c = d[-1]
    for i in range(len(d) - 2, -1, -1):
        c = cv2.pyrUp(c, dstsize=(d[i].shape[1], d[i].shape[0])) + d[i]
    return c


def main():
    a = cv2.imread(D + '1_1.jpg')
    b = cv2.imread(D + '1_2.jpg')
    p1, p2 = hw.feature_matching(a, b)
    H, m, cnt = utils.ransac_with_inliers(p1, p2, hw.compute_homography,
                                          n_iter=1000, tol=5.0)

    h1, w1 = a.shape[:2]
    h2, w2 = b.shape[:2]

    # 先算画布多大，跟框架里 stitch_blend 的做法一样
    pt = []
    for cx, cy in ((0, 0), (0, h1), (w1, 0), (w1, h1)):
        q = H.dot(np.array([cx, cy, 1.0]))
        pt.append(q / q[2])
    x0 = int(min(0, *(p[0] for p in pt)))
    x1 = int(max(w2, *(p[0] for p in pt)))
    y0 = int(min(0, *(p[1] for p in pt)))
    y1 = int(max(h2, *(p[1] for p in pt)))

    x, y = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    x = np.float32(x)
    y = np.float32(y)

    # 把两张图摆到同一块画布上，顺手记下各自盖住了哪儿
    M = np.linalg.pinv(H)
    z = M[2, 0] * x + M[2, 1] * y + M[2, 2]
    tx = np.float32((M[0, 0] * x + M[0, 1] * y + M[0, 2]) / z)
    ty = np.float32((M[1, 0] * x + M[1, 1] * y + M[1, 2]) / z)
    c1 = cv2.remap(np.float32(a), tx, ty, cv2.INTER_LINEAR)
    c2 = cv2.remap(np.float32(b), x, y, cv2.INTER_LINEAR)
    w1 = cv2.remap(np.ones((h1, w1), np.float32), tx, ty, cv2.INTER_LINEAR)
    w2 = cv2.remap(np.ones((h2, w2), np.float32), x, y, cv2.INTER_LINEAR)

    # 边框被黑边掺过，颜色先除回去，不然接缝会暗一截
    c1 = c1 / np.maximum(w1[..., None], 1e-6)
    c2 = c2 / np.maximum(w2[..., None], 1e-6)
    s = w1 + w2
    g0 = np.where(s > 1e-6, w1 / np.maximum(s, 1e-6), 0.5)

    # 直接按掩码平均，这是对照的那版
    base = (c1 * w1[..., None] + c2 * w2[..., None]) / np.maximum(s[..., None], 1e-6)

    # 低频慢慢过渡，高频各留各的
    d1 = lap(c1)
    d2 = lap(c2)
    g = [g0]
    for i in range(len(d1) - 1):
        g.append(cv2.pyrDown(g[-1]))
    mix = []
    for i in range(len(d1)):
        q = g[i]
        if q.shape[:2] != d1[i].shape[:2]:
            q = cv2.resize(q, (d1[i].shape[1], d1[i].shape[0]))
        mix.append(d1[i] * q[..., None] + d2[i] * (1 - q[..., None]))
    multi = back(mix)

    # 边上一圈谁也没盖到的黑块裁掉，好看清接缝
    ok = s > 0.9
    ys, xs = np.nonzero(ok)
    r0, r1 = int(ys.min()), int(ys.max() + 1)
    q0, q1 = int(xs.min()), int(xs.max() + 1)
    u = np.uint8(np.clip(base[r0:r1, q0:q1], 0, 255))
    v = np.uint8(np.clip(multi[r0:r1, q0:q1], 0, 255))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].imshow(rgb(u))
    ax[0].set_title('(a) weighted average')
    ax[0].axis('off')
    ax[1].imshow(rgb(v))
    ax[1].set_title('(b) Laplacian pyramid blending')
    ax[1].axis('off')
    fig.savefig('outputs/fig_bonus.png', dpi=140, bbox_inches='tight')
    plt.close(fig)
    print('画布', x1 - x0 + 1, 'x', y1 - y0 + 1, '| 内点', cnt, '/', len(p1),
          '| 裁剪后', u.shape[1], 'x', u.shape[0], '| 图已写出')


if __name__ == '__main__':
    main()
