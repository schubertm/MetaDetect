#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import numpy as np
import pandas as pd
from src.bbox_tools.iou_metrics import bbox_iou

def old_nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes

def nms_temp(box, bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param box/bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    best_bboxes = []

    # bboxes = np.delete(bboxes, index, 0)

    cls_mask = (bboxes[:, 5].astype(int) == int(box[0, 5]))
    # print(cls_mask)
    cls_bboxes = bboxes[cls_mask, :]

    # while len(cls_bboxes) > 0:
    best_bbox = np.asarray(box).flatten()
    best_bboxes.append(best_bbox)
    # cls_bboxes = np.concatenate([cls_bboxes[: index], cls_bboxes[index + 1:]])
    iou = bbox_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
    # print(np.max(iou))
    weight = np.zeros((len(iou),), dtype=np.float32)

    assert method in ['nms', 'soft-nms']

    if method == 'nms':
        iou_mask = iou > iou_threshold
        weight[iou_mask] = 1.0

    if method == 'soft-nms':
        weight = np.exp(-(1.0 * iou ** 2 / sigma))

    cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
    score_mask = (weight == 1.)
    cls_bboxes = cls_bboxes[score_mask]
    iou = iou[score_mask]

    iou[iou == 1] = 0

    # print(cls_bboxes)
    cls_bboxes = np.concatenate((np.asmatrix(cls_bboxes), np.transpose(np.asmatrix(iou))), axis=1)

    return cls_bboxes

def perform_nms_on_dataframe(df):
    img_paths = list(set(df["file_path"]))
    post_nms_df_list = []
    for p in img_paths:
        # img_id = 0
        img_df = df[df["file_path"].isin([p])] #.drop("gradient_metrics", axis=1)
        cols = list(img_df.loc[:, "xmin":].columns)
        img_post_nms = old_nms(np.array(img_df.loc[:, "xmin":]), 0.5)
        post_nms_frame = pd.DataFrame(data=img_post_nms, columns=cols)
        post_nms_frame['file_path'] = p
        post_nms_frame['dataset_box_id'] = np.arange(0, len(post_nms_frame), 1)
        post_nms_frame = post_nms_frame[['file_path', 'dataset_box_id'] + cols]
        post_nms_df_list.append(post_nms_frame)

    pnms_df = pd.concat(post_nms_df_list, axis=0, ignore_index=True).sort_values(by="dataset_box_id")
    pnms_df.index = range(len(pnms_df))

    return pnms_df


def select_rows_from_other(df, ids_df, by="dataset_box_id"):
    id_list = list(ids_df[by])
    df = df[df[by].isin(id_list)]

    return df.sort_values(by=by)

