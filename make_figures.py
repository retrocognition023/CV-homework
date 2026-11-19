# -*- coding: utf-8 -*-
"""给报告画图的脚本。跑完往 outputs/ 里写若干张 fig_*.png。"""
import os
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
import homework2 as hw
import utils

os.makedirs('outputs', exist_ok=True)
D = 'Problem2Images/'


def rgb(a):
    return cv2.cvtColor(a, cv2.COLOR_BGR2RGB)


def save(fig, name):
    fig.savefig('outputs/' + name, dpi=140, bbox_inches='tight')
    plt.close(fig)


def fig_corners():
    """Q1：Harris 角点 + 响应图。"""
    a = cv2.imread(D + '1_1.jpg')
    R = hw.harris_response(a, hw.HARRIS_K, hw.HARRIS_WIN)
    c = hw.corner_selection(R, hw.CORNER_THRESH * np.max(R),
                            hw.CORNER_MIN_DIST)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].imshow(rgb(a))
    ax[0].plot([p[0] for p in c], [p[1] for p in c], 'r+', markersize=4)
    ax[0].set_title('Harris corners on 1_1.jpg (%d points)' % len(c))
    ax[0].axis('off')
    lim = np.percentile(np.abs(R), 99)
    ax[1].imshow(R, cmap='RdBu_r', vmin=-lim, vmax=lim)
    ax[1].set_title('Response R (k=0.04, window=9)')
    ax[1].axis('off')
    save(fig, 'fig_corners.png')


def fig_matches(pair, name, title):
    """Q2/Q3：把两图并排连线；绿色=RANSAC 内点，橙色=被剔除。"""
    a = cv2.imread(D + pair[0])
    b = cv2.imread(D + pair[1])
    p1, p2 = hw.feature_matching(a, b)
    H, m, cnt = utils.ransac_with_inliers(p1, p2, hw.compute_homography,
                                          n_iter=1000, tol=5.0)
    h1, w1 = a.shape[:2]
    h2, w2 = b.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = a
    canvas[:h2, w1:] = b
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(rgb(canvas))
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(p1, p2)):
        ax.plot([x1, x2 + w1], [y1, y2],
                color=('#2d8a4e' if m[i] else '#e07b39'),
                linewidth=0.8, alpha=0.85)
    ax.set_title('%s   raw=%d, inliers=%d (%.1f%%)' %
                 (title, len(p1), cnt, 100.0 * cnt / len(p1)))
    ax.axis('off')
    save(fig, name)
    return len(p1), cnt


def fig_inliers():
    """Q4.2：把被 RANSAC 删掉的匹配单独标红，附统计。"""
    a = cv2.imread(D + '2_1.jpg')
    b = cv2.imread(D + '2_2.jpg')
    p1, p2 = hw.feature_matching(a, b)
    H, m, cnt = utils.ransac_with_inliers(p1, p2, hw.compute_homography,
                                          n_iter=1000, tol=5.0)
    h1, w1 = a.shape[:2]
    h2, w2 = b.shape[:2]
    canvas = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = a
    canvas[:h2, w1:] = b
    fig, ax = plt.subplots(figsize=(18, 6))
    ax.imshow(rgb(canvas))
    for i, ((x1, y1), (x2, y2)) in enumerate(zip(p1, p2)):
        if not m[i]:
            ax.plot([x1, x2 + w1], [y1, y2], color='#e07b39',
                    linewidth=1.2, alpha=0.9)
    ax.set_title('RANSAC on pair 2: %d/%d kept, %d rejected (orange)' %
                 (cnt, len(p1), len(p1) - cnt))
    ax.axis('off')
    save(fig, 'fig_inliers.png')
    e = utils.reproj_err(H, p1, p2)
    with open('outputs/ransac_stats.txt', 'w', encoding='utf-8') as f:
        f.write('pair=2_1/2_2\nraw_matches=%d\ninliers=%d\n'
                'inlier_ratio=%.4f\nrejected=%d\ntol=5.0px\n'
                'err_median_inlier=%.3f\nerr_max_inlier=%.3f\n' %
                (len(p1), cnt, cnt / len(p1), len(p1) - cnt,
                 np.median(e[m]) if cnt else -1,
                 e[m].max() if cnt else -1))
    return len(p1), cnt


def fig_hyper():
    """Q6 实验一：Harris 的 k 对角点数量的影响。"""
    a = cv2.imread(D + '1_1.jpg')
    b = cv2.imread(D + '1_2.jpg')
    rows = []
    for k in (0.02, 0.04, 0.06, 0.08):
        R1 = hw.harris_response(a, k, hw.HARRIS_WIN)
        R2 = hw.harris_response(b, k, hw.HARRIS_WIN)
        c1 = hw.corner_selection(R1, hw.CORNER_THRESH * np.max(R1),
                                 hw.CORNER_MIN_DIST)
        c2 = hw.corner_selection(R2, hw.CORNER_THRESH * np.max(R2),
                                 hw.CORNER_MIN_DIST)
        rows.append((k, len(c1), len(c2)))
    fig, ax = plt.subplots(figsize=(8, 4))
    xx = np.arange(len(rows))
    ax.bar(xx - 0.18, [r[1] for r in rows], 0.36, label='corners in 1_1')
    ax.bar(xx + 0.18, [r[2] for r in rows], 0.36, label='corners in 1_2')
    ax.set_xticks(xx)
    ax.set_xticklabels(['k=%.2f' % r[0] for r in rows])
    ax.set_ylabel('number of corners')
    ax.set_title('Q6 experiment: effect of Harris constant k')
    ax.legend()
    save(fig, 'fig_exp_k.png')
    with open('outputs/exp_k.txt', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('k=%.2f  img1=%d  img2=%d\n' % r)


def fig_ratio():
    """Q6 实验二：ratio-test 阈值对匹配数与拼接成功的影响。"""
    a = cv2.imread(D + '1_1.jpg')
    b = cv2.imread(D + '1_2.jpg')
    rows = []
    old = hw.MATCH_RATIO
    for q in (0.60, 0.65, 0.70, 0.75):
        hw.MATCH_RATIO = q
        p1, p2 = hw.feature_matching(a, b)
        H, m, cnt = utils.ransac_with_inliers(p1, p2, hw.compute_homography,
                                              n_iter=500, tol=5.0)
        rows.append((q, len(p1), cnt))
    hw.MATCH_RATIO = old
    fig, ax = plt.subplots(figsize=(8, 4))
    xx = np.arange(len(rows))
    ax.bar(xx - 0.18, [r[1] for r in rows], 0.36, label='raw matches')
    ax.bar(xx + 0.18, [r[2] for r in rows], 0.36, label='RANSAC inliers')
    ax.set_xticks(xx)
    ax.set_xticklabels(['ratio=%.2f' % r[0] for r in rows])
    ax.set_ylabel('count')
    ax.set_title('Q6 experiment: effect of ratio-test threshold')
    ax.legend()
    save(fig, 'fig_exp_ratio.png')
    with open('outputs/exp_ratio.txt', 'w', encoding='utf-8') as f:
        for r in rows:
            f.write('ratio=%.2f  raw=%d  inliers=%d\n' % r)


if __name__ == '__main__':
    t = time.time()
    fig_corners()
    print('corners ok')
    print('match pair1:', fig_matches(('1_1.jpg', '1_2.jpg'),
                                     'fig_match_1.png',
                                     'Pair 1 (translation)'))
    print('match pair4:', fig_matches(('4_1.jpg', '4_2.jpg'),
                                     'fig_match_4.png',
                                     'Pair 4 (rotation + scale)'))
    print('inliers pair2:', fig_inliers())
    fig_hyper()
    fig_ratio()
    print('experiments ok, %.1fs' % (time.time() - t))
