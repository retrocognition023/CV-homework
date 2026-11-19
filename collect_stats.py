# -*- coding: utf-8 -*-
# 收集报告要用的真实统计数字，结果存成 stats.json
import json
import cv2
import numpy as np
import homework2 as hw
import utils

out = {}
for tag, pair in [('1', ('1_1.jpg', '1_2.jpg')),
                  ('2', ('2_1.jpg', '2_2.jpg')),
                  ('3', ('3_1.JPG', '3_2.JPG')),
                  ('4', ('4_1.jpg', '4_2.jpg'))]:
    a = cv2.imread('Problem2Images/' + pair[0])
    b = cv2.imread('Problem2Images/' + pair[1])
    p1, p2 = hw.feature_matching(a, b)
    H, m, c = utils.ransac_with_inliers(p1, p2, hw.compute_homography,
                                        n_iter=1000, tol=5.0)
    e = utils.reproj_err(H, p1, p2)
    out['pair_' + tag] = {
        'raw': len(p1), 'inliers': c,
        'ratio': round(100.0 * c / len(p1), 1),
        'err_med': round(float(np.median(e[m])), 3) if c else -1,
        'err_max': round(float(e[m].max()), 3) if c else -1,
        'H': H.tolist(),
    }

# 四个场景逐对匹配数
scenes = {}
for sid, scene in hw.PANORAMA_SCENES.items():
    ims = hw.load_image_sequence('Problem2Images/panoramas/' + scene)
    if len(ims) > hw.BASE_MAX_FRAMES:
        mid = len(ims) // 2
        half = hw.BASE_MAX_FRAMES // 2
        ims = ims[mid - half:mid - half + hw.BASE_MAX_FRAMES]
    rows = []
    for j in range(len(ims) - 1):
        try:
            p1, p2 = hw.feature_matching(ims[j], ims[j + 1])
            H, m, c = utils.ransac_with_inliers(p1, p2, hw.compute_homography,
                                                n_iter=800, tol=5.0)
            rows.append({'pair': j, 'raw': len(p1), 'inliers': c})
        except Exception as ex:
            rows.append({'pair': j, 'raw': -1, 'inliers': -1})
    scenes[scene] = rows
out['scenes'] = scenes

with open('outputs/stats.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=1)
print(json.dumps(out, ensure_ascii=False)[:1600])
