#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from api.pre_process import standardize_columns
from src.bbox_tools.nms_algorithms import nms_temp
from configs.uq_config import OUTPUT_METRICS
import configs.data_config as data_cfg

#todo: do we need this here?
def add_meta_detect_metrics(variables_df, metrics_const):
    if "meta_detect" in metrics_const:
        if os.path.exists(data_cfg.default_df_path + '/md_metrics_0.0.csv'):
            md_df = pd.read_csv(data_cfg.default_df_path + '/md_metrics_0.0.csv').drop('Unnamed: 0', axis=1)
        else:
            md_df = output_based_metrics(variables_df[["file_path"] + OUTPUT_METRICS], 0.0)
    #     grads_df = variables_df[[s for s in variables_df.columns if "grad" in s]]
    #     variables_df = pd.concat([md_df, grads_df], axis=1)
    # return standardize_columns(md_df.astype(float))
    return md_df

def get_nms_dict(metrics_matrix, score_threshold=0.3):
    # save all metrics in one .CSV file
    header_str = data_cfg.META_DETECT_METRICS
    target_path = "{}/md_metrics_{:.2}.csv".format(data_cfg.default_df_path, score_threshold)
    # pd.DataFrame(metrics_matrix).to_csv("/home/schubert/file_score=" + str(score_threshold) + ".csv", header=header_str)
    md_metrics = pd.DataFrame(metrics_matrix)
    md_metrics.to_csv(target_path, header=header_str)

    return md_metrics


def get_nms_dict_temp(bboxes):
    bboxes2 = np.asmatrix(bboxes.copy())
    metrics_single_image = np.zeros((bboxes2.shape[0], 40))
    for i in range(bboxes2.shape[0]):
        minimum = 0
        maximum = 0
        candidates = nms_temp(np.asmatrix(bboxes2[i, :]), bboxes, 0.45)
        candidates = np.concatenate((candidates[:, :5], np.asmatrix(candidates[:, -1])), axis=1)
        assignment_bboxes = np.zeros((candidates.shape[0], 9))
        for j in range(candidates.shape[0]):
            assignment_bboxes[j, 0] = 0.5 * (candidates[j, 0] + candidates[j, 2])
            assignment_bboxes[j, 1] = 0.5 * (candidates[j, 1] + candidates[j, 3])
            assignment_bboxes[j, 2] = abs(0.5 * (candidates[j, 0] - candidates[j, 2]))
            assignment_bboxes[j, 3] = abs(0.5 * (candidates[j, 1] - candidates[j, 3]))
            assignment_bboxes[j, 4] = candidates[j, 4]
            assignment_bboxes[j, 5] = candidates[j, 5]
            assignment_bboxes[j, 6] = (0.5 * ((0.5 * (candidates[j, 0] + candidates[j, 2])) - ( abs(0.5 * (candidates[j, 0] - candidates[j, 2]))))) * (0.5 * ((0.5 * (candidates[j, 1] + candidates[j, 3])) - (abs(0.5 * (candidates[j, 1] - candidates[j, 3])))))
            assignment_bboxes[j, 7] = (candidates[j, 2] - candidates[j, 0]) + (candidates[j, 3] - candidates[j, 1])
            assignment_bboxes[j, 8] = ((0.5 * (abs(candidates[j, 2] - candidates[j, 0]))) * (0.5 * (abs(candidates[j, 3] - candidates[j, 1])))) / ((abs(candidates[j, 2] - candidates[j, 0])) + (abs(candidates[j, 3] - candidates[j, 1])))

        iou = ((np.asarray(assignment_bboxes[:, 5])).flatten()).astype('float32')

        if len(iou) > 1:
            minimum = np.min(iou[iou > 0])
            maximum = np.max(iou[:])

        assignment_bboxes = assignment_bboxes.astype('float32')

        metrics_single_image[i, :] = [assignment_bboxes.shape[0], float(np.min((assignment_bboxes[:, 0]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 0]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 0]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 0]), axis=0)),
                                      float(np.min((assignment_bboxes[:, 1]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 1]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 1]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 1]), axis=0)),
                                      float(np.min((assignment_bboxes[:, 2]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 2]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 2]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 2]), axis=0)),
                                      float(np.min((assignment_bboxes[:, 3]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 3]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 3]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 3]), axis=0)),
                                      float((0.5 * ((0.5 * (bboxes2[i, 0] + bboxes2[i, 2])) - (abs(0.5 * (bboxes2[i, 0] - bboxes2[i, 2]))))) * (0.5 * ((0.5 * (bboxes2[i, 1] + bboxes2[i, 3])) - (abs(0.5 * (bboxes2[i, 1] - bboxes2[i, 3])))))),
                                      float(np.min((assignment_bboxes[:, 6]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 6]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 6]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 6]), axis=0)),
                                      float((abs(bboxes2[i, 2] - bboxes2[i, 0])) + (abs(bboxes2[i, 3] - bboxes2[i, 1]))),
                                      float(np.min((assignment_bboxes[:, 7]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 7]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 7]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 7]), axis=0)),
                                      float(((0.5 * (abs(bboxes2[i, 2] - bboxes2[i, 0]))) * (0.5 * (abs(bboxes2[i, 3] - bboxes2[i, 1])))) / ((abs(bboxes2[i, 2] - bboxes2[i, 0])) + (abs(bboxes2[i, 3] - bboxes2[i, 1])))),
                                      float(np.min((assignment_bboxes[:, 8]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 8]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 8]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 8]), axis=0)),
                                      float(np.min((assignment_bboxes[:, 4]), axis=0)),
                                      float(np.max((assignment_bboxes[:, 4]), axis=0)),
                                      float(np.mean((assignment_bboxes[:, 4]), axis=0)),
                                      float(np.std((assignment_bboxes[:, 4]), axis=0)), float(minimum), float(maximum),
                                      float(max(0, np.mean(iou[iou > 0]))), float(max(0, np.std(iou[iou > 0])))]

    return metrics_single_image


def output_based_metrics(metrics_frame, score_threshold):
    # metrics_frame = load_single_frame('/home/riedlinger/MetaDetect-TestEvaluation/grads_lr=1e-3_bs=64/metrics_cat.csv')

    bboxes = metrics_frame.loc[:, 'file_path':'category_idx']
    bboxes = bboxes.loc[bboxes['s'] > score_threshold]
    bboxes = bboxes.to_numpy()

    all_image_paths = list(set(metrics_frame["file_path"]))
    pbar = tqdm(total=len(all_image_paths))
    #todo: make while into for with list(set(...))

    for i in range(len(all_image_paths)):

        box_mask = (bboxes[:, 0] == all_image_paths[i]).T
        bboxes_single_image = bboxes[box_mask, 1:]

        metrics_single_image = get_nms_dict_temp(bboxes_single_image)

        try:
            metrics_matrix = np.append(metrics_matrix, metrics_single_image, axis=0)
        except:
            metrics_matrix = metrics_single_image

        pbar.update(1)

    pbar.close()
    return get_nms_dict(metrics_matrix, score_threshold)


# if __name__ == '__main__':
#     var_df, targets_df = load_or_prepare_metrics(data_cfg.default_df_path, "logistic", ["output"])
#     output_based_metrics(var_df, .0)
