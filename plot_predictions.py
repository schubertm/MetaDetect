import cv2
import pandas as pd
import numpy as np
import random
import os
import colorsys
from src.bbox_tools.nms_algorithms import perform_nms_on_dataframe
import configs.data_config as data_cfg
import xgboost as xgb
from matplotlib import pyplot as plt
from matplotlib import colors as col
from matplotlib import cm as cmx


def draw_bbox_fill(image, bboxes, classes=data_cfg.CLASS_NAMES, show_label=True, gt=True, fill=False, predicted=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, cls_id] format coordinates.
    """
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / (num_classes + 1), 1., 1.) for x in range(num_classes + 1)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    if fill == True:
        jet = plt.get_cmap("RdYlGn")
        # cNorm = col.DivergingNorm(vcenter=0.5, vmin=0, vmax=1)
        cNorm = col.Normalize(vmin=0, vmax=1)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

    for i, bbox in enumerate(bboxes):
        image2 = image.copy()
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        # fontScale = 1
        score = bbox[4]
        class_ind = int(bbox[5])
        # print(bboxes)
        if len(bbox) == 7:
            predicted_iou = bbox[6]
        if len(bbox) == 8:
            true_iou = bbox[6]
            predicted_iou = bbox[7]
        if fill == False:
            bbox_color = colors[class_ind]
            if gt == True:
                bbox_color = colors[np.asarray(colors).shape[0] - 1]
                # print('gt = ' +str(bbox_color))
        else:
            bbox_color = scalarMap.to_rgba(predicted_iou)[:3]
            bbox_color = [255 * i for i in bbox_color]
            bbox_color = bbox_color[::-1]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        bbox_thick = 3 * int(0.6 * (image_h + image_w) / 600)
        if fill == True:
            bbox_thick = -1
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        alpha = 0.55
        image = cv2.addWeighted(image, alpha, image2, 1 - alpha, 0)

        if show_label and fill == False:
            try:
                try:
                    bbox_mess = '%s: / %.2f / %.2f' % (classes[class_ind], score, true_iou)
                except:
                    bbox_mess = '%s: %.2f / %.2f' % (classes[class_ind], score, predicted_iou)
            except:
                bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            # bbox_mess_class = '%s' % (classes[class_ind])

            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=int(0.6 * (image_h + image_w) / 600) // 2)[0]
            # t2_size = cv2.getTextSize(bbox_mess_class, 0, fontScale, thickness=bbox_thick//2)[0]

            cv2.rectangle(image, (c1[0], c1[1]), (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
            # cv2.rectangle(image, (c1[0],c2[1]), (c1[0] + t2_size[0], c2[1] + t2_size[1] + 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), int(0.6 * (image_h + image_w) / 600) // 2, lineType=cv2.LINE_AA)

            # cv2.putText(image, bbox_mess_class, (c1[0], c2[1]+6), cv2.FONT_HERSHEY_SIMPLEX,
            # fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    if fill == True:
        line, column, _ = np.shape(image)
        if predicted == True:
            cv2.putText(image, 'predicted iou', (int(column - 120), 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        (255, 255, 255), int(1.25 * (image_h + image_w) / 600) // 2, lineType=cv2.LINE_AA)
        else:
            cv2.putText(image, 'true iou', (int(column - 120), 20), cv2.FONT_HERSHEY_SIMPLEX, fontScale,
                        (255, 255, 255), int(1.25 * (image_h + image_w) / 600) // 2, lineType=cv2.LINE_AA)

    return image


def draw_bbox(image, bboxes, classes=data_cfg.CLASS_NAMES, show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image


def plot_preds(data_path, save_path):

    if not os.path.exists(save_path): os.makedirs(save_path)

    df = pd.read_csv(data_path + 'metrics_cat.csv').drop(['Unnamed: 0', 'dataset_box_id'], axis=1)
    df = perform_nms_on_dataframe(df)
    image_paths = list(set(df['file_path']))

    for i in range(len(image_paths)):
        df_temp = df.loc[df['file_path'] == image_paths[i]]
        image = cv2.imread(image_paths[i])
        # print(df_temp[['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx']])
        image = draw_bbox(image, np.asarray(df_temp[['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx']]))
        cv2.imwrite(save_path + image_paths[i].split('/')[-1], image)

def plot_preds_with_iou(data_path, save_path):

    if not os.path.exists(save_path): os.makedirs(save_path)

    df = pd.read_csv(data_path + 'metrics_cat.csv').drop(['Unnamed: 0', 'dataset_box_id'], axis=1)
    df_metrics = pd.read_csv(data_path + 'md_metrics_0.0.csv').drop('Unnamed: 0', axis=1)
    df = pd.concat([df, df_metrics], axis=1)
    iou = pd.read_csv(data_path + 'true_iou.csv').drop('Unnamed: 0', axis=1)
    df['true_iou'] = iou['true_iou'].astype(float)
    df = perform_nms_on_dataframe(df)
    image_paths = list(set(df['file_path']))

    random.seed(0)
    random.shuffle(image_paths)

    split_1 = image_paths[:int(len(image_paths) / 2)]
    split_2 = image_paths[int(len(image_paths) / 2):]

    df_1 = df[df['file_path'].isin(split_1)]
    df_2 = df[df['file_path'].isin(split_2)]

    model1 = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=2, use_label_encoder=False, max_depth=3, n_estimators=29, reg_alpha=1.5, reg_lambda=0.0, learning_rate=0.3)
    model2 = xgb.XGBClassifier(tree_method="gpu_hist", gpu_id=2, use_label_encoder=False, max_depth=3, n_estimators=29, reg_alpha=1.5, reg_lambda=0.0, learning_rate=0.3)

    model1.fit(df_1.drop(['file_path', 'dataset_box_id', 'true_iou'], axis=1), df_1['true_iou'].round(0))
    model2.fit(df_2.drop(['file_path', 'dataset_box_id', 'true_iou'], axis=1), df_2['true_iou'].round(0))

    iou_1 = model1.predict_proba(df_2.drop(['file_path', 'dataset_box_id', 'true_iou'], axis=1))[:, 1]
    iou_2 = model2.predict_proba(df_1.drop(['file_path', 'dataset_box_id', 'true_iou'], axis=1))[:, 1]

    df_2['end_score'] = iou_1
    df_1['end_score'] = iou_2

    df_post = pd.concat([df_2, df_1], ignore_index=True, axis=0)

    for i in range(len(image_paths)):
        df_temp = df_post.loc[df_post['file_path'] == image_paths[i]]
        image = cv2.imread(image_paths[i])
        image2 = image.copy()

        image_temp = draw_bbox_fill(image, np.asarray(df_temp[['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx', 'end_score']]), show_label=True, gt=False)
        image = draw_bbox_fill(image_temp, np.asarray(df_temp[['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx', 'end_score']]), show_label=True, gt=False, fill=True)

        image_temp2 = draw_bbox_fill(image2, np.asarray(df_temp[['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx', 'true_iou']]), show_label=True, gt=False)
        image2 = draw_bbox_fill(image_temp2, np.asarray(df_temp[['xmin', 'ymin', 'xmax', 'ymax', 's', 'category_idx', 'true_iou']]), show_label=True, gt=False, fill=True, predicted=False)

        image = np.concatenate((image, image2), axis=0)

        cv2.imwrite(save_path + image_paths[i].split('/')[-1], image)


def plot_single():
    data_path = data_cfg.default_df_path + '/'
    save_path = str(data_path) + '/images/predictions/'
    plot_preds(data_path, save_path)

def plot_iou():
    data_path = data_cfg.default_df_path + '/'
    save_path = str(data_path) + '/images/ious/'
    plot_preds_with_iou(data_path, save_path)


if __name__ == '__main__':
    plot_single()
    plot_iou()
