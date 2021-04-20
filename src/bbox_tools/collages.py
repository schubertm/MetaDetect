#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import cv2
import os
from src.bbox_tools.nms_algorithms import old_nms

pred_path = "/home/riedlinger/MetaDetect-TestEvaluation/metrics_cat.csv"
iou_path = "/home/riedlinger/MetaDetect-TestEvaluation/true_iou.csv"

score_thresh = 0.3

pred_df = pd.read_csv(pred_path)
iou_df = pd.read_csv(iou_path)

path_list = list(set(pred_df["file_path"]))

for p in tqdm.tqdm(path_list):
    has_bad_sample = False
    img_id = p.split("/")[-1].split(".")[0]
    img_pred_df = pred_df[pred_df["file_path"].isin([p])]
    img_iou_df = iou_df[iou_df["dataset_box_id"].isin(list(img_pred_df["dataset_box_id"]))]

    gt_df = pd.read_csv(f"/home/riedlinger/dataset_ground_truth/KITTI/csv/{img_id}_gt.csv")

    base_img = cv2.imread(p)

    gt_img = base_img.copy()
    coords = ["xmin", "ymin", "xmax", "ymax"]

    for i in gt_df.index:
        b = gt_df.loc[i, coords]
        cv2.rectangle(gt_img, (b[0], b[1]), (b[2], b[3]), color=(0, 165, 255), thickness=2)

    data_df = pd.concat([img_pred_df, img_iou_df], axis=1)
    data_df = data_df[data_df["s"] >= score_thresh]

    pred_data = data_df[coords+["s", "category_idx", "true_iou"]]

    pred = old_nms(np.array(pred_data), iou_threshold=0.45)
    pred_img = base_img.copy()
    for b in pred:
        cv2.rectangle(pred_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), color=(200, 255, 200), thickness=1)

    score_img = pred_img.copy()
    # iou_img = pred_img.copy()

    for b in pred:
        if b[6] < 0.1:
            # print(img_id)
            c = (200, 200, 255)
            has_bad_sample = True
        else:
            c = (200, 255, 200)
        s_str = "s={:.2}/IoU={:.2}".format(b[4], b[6])
        t_size = cv2.getTextSize(s_str, 0, 0.5, thickness=1)[0]
        cv2.rectangle(score_img, (int(b[0]), int(b[1]) - t_size[1] - 1), (int(b[0]) + t_size[0], int(b[1])),
                      c, -1)
        cv2.putText(score_img, s_str, (int(b[0]), int(b[1]) - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
                    color=(0, 0, 0), thickness=1)

        # iou_str = "IoU = {:.3}".format(b[6])
        # t_size = cv2.getTextSize(iou_str, 0, 0.35, thickness=1)[0]
        # cv2.rectangle(iou_img, (int(b[0]), int(b[1]) - t_size[1]), (int(b[0]) + t_size[0], int(b[1])),
        #               (200, 255, 200), -1)
        # cv2.putText(iou_img, s_str, (int(b[0]), int(b[1]) - 1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5,
        #             color=(0, 0, 0), thickness=1)
    if has_bad_sample:
        coll = np.vstack([gt_img, score_img])
        cv2.imwrite(f"/home/riedlinger/MetaDetect-TestEvaluation/prediction_collages/pred_collage_{img_id}.png", coll)
