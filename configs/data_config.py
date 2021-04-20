#  Copyright 2020  Marius Schubert, Tobias Riedlinger
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
#  1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
#  2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
#  3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
#
default_df_path = "/home/schubert/grads_lr=1e-3_bs=64"

default_gt_path = "/home/schubert/datasets_ground_truth/KITTI/csv"

num_classes = 8

CLASS_NAMES = ['car', 'van', 'truck', 'pedestrian', 'person', 'cyclist', 'tram', 'misc']

SCORE_THRESHOLDS_META_DETECT = [0.01, 0.1, 0.3, 0.5]

PERFORM_NMS = True

# # #
#       Metrics identifiers used in DataFrame
# # #
OUTPUT_METRICS = ["xmin", "ymin", "xmax", "ymax", "s", "category_idx", "prob_sum"] + [f"prob_{i}" for i in range(num_classes)]
STD_OUTPUT_METRICS = ["s", "prob_sum"] + [f"prob_{i}" for i in range(num_classes)]
META_DETECT_METRICS = ['Number of Candidate Boxes', 'x_min',
                  'x_max', 'x_mean', 'x_std', 'y_min', 'y_max', 'y_mean', 'y_std', 'w_min', 'w_max', 'w_mean', 'w_std',
                  'h_min', 'h_max', 'h_mean', 'h_std', 'size', 'size_min', 'size_max', 'size_mean', 'size_std',
                  'circum', 'circum_min', 'circum_max', 'circum_mean', 'circum_std', 'size/circum', 'size/circum_min',
                  'size/circum_max', 'size/circum_mean', 'size/circum_std', 'score_min', 'score_max', 'score_mean', 'score_std',
                  'IoU_pb_min', 'IoU_pb_max', 'IoU_pb_mean', 'IoU_pb_std']
