import os
import random
import numpy as np
import argparse
from base_functions import load_img_by_gdal
from sklearn.metrics import confusion_matrix

SMOOTH = 0.001

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description='RS classification train')
    parser.add_argument('--gt', dest='ref_file', help='ground truth',
                        default='D:\\data\\evaluate\\gt\\ZY304016220151108.tif')
    parser.add_argument('--mask', dest='pred_file', help='predict mask',
                        default='D:\\data\\evaluate\\mask\\ZY304016220151108.tif')
    parser.add_argument('--maxrange', dest='maxrange', help='max value of range',
                        default='6')
    parser.add_argument('--rate', dest='check_rate', help='check rate',
                        default='1.0')
    args = parser.parse_args()

    ref_file = args.ref_file
    pred_file = args.pred_file
    valid_labels = range(int(args.maxrange))
    n_class = len(valid_labels)
    check_rate = float(args.check_rate)


    ref_img = load_img_by_gdal(ref_file, grayscale=True)

    pred_img = load_img_by_gdal(pred_file, grayscale=True)

    print("\nfile: {}".format(os.path.split(pred_file)[1]))

    print("[INFO] Calculate confusion matrix..\n")

    ref_img = np.array(ref_img)
    height, width = ref_img.shape
    print(height, width)
    if height != pred_img.shape[0] or width != pred_img.shape[1]:
        print("image sizes of reference and predicted are not equal!\n")

    img_length = height * width
    if check_rate<0.01:
        check_rate=0.01
    elif check_rate>1.00:
        check_rate=1.00
    else:
        pass

    # assert (check_rate > 0.001 and check_rate <= 1.00)
    num_checkpoints = np.int(img_length * check_rate)

    pos = random.sample(range(img_length), num_checkpoints)

    """reshape images from two to one dimension"""
    ref_img = np.reshape(ref_img, height * width)
    pred_img = np.reshape(pred_img, height * width)

    labels = ref_img[pos]

    """ignore nodata pixels"""
    valid_index = []
    for tt in valid_labels:
        ind = np.where(labels == tt)
        ind = list(ind)
        valid_index.extend(ind[0])

    valid_index.sort()
    valid_num_checkpoints = len(valid_index)
    print("{}points have been selected, but {} points will be used to evaluate accuracy!\n".format(num_checkpoints,
                                                                                                   valid_num_checkpoints))
    # valid_ref=ref_img[valid_index]
    valid_ref = labels[valid_index]
    print("valid value in reference image: {}".format(np.unique(valid_ref)))

    ts = pred_img[pos]
    valid_pred = ts[valid_index]
    print("valid value in predicton image: {}".format(np.unique(valid_pred)))

    tmp = np.unique(valid_pred)
    """make nodata to zero in predicted results"""
    for ss in tmp:
        if not ss in valid_labels:
            nodata_index = np.where(valid_pred==ss)
            valid_pred[nodata_index]=0

    confus_matrix = confusion_matrix(valid_pred, valid_ref, range(n_class))

    print(confus_matrix)

    confus_matrix = np.array(confus_matrix)

    oa = 0
    x_row_plus = []
    x_col_plus = []
    x_diagonal = []
    for i in range(n_class):
        oa += confus_matrix[i, i]
        x_diagonal.append(confus_matrix[i, i])
        row_sum = sum(confus_matrix[i, :])
        col_sum = sum(confus_matrix[:, i])
        x_row_plus.append(row_sum)
        x_col_plus.append(col_sum)

    print("x_row_plus:{}".format(x_row_plus))
    print("x_col_plus:{}".format(x_col_plus))
    x_row_plus = np.array(x_row_plus)
    x_col_plus = np.array(x_col_plus)
    x_diagonal = np.array(x_diagonal)
    x_total = sum(x_row_plus)
    OA_acc = oa / (sum(x_row_plus))
    print("\nOA:{:.3f}".format(OA_acc))
    tmp = x_col_plus * x_row_plus
    kappa = (x_total * sum(x_diagonal) - sum(x_col_plus * x_row_plus)) / np.float(x_total * x_total - sum(x_col_plus * x_row_plus))

    print("Kappa:{:.3f}".format(kappa))

    for i in range(n_class):
        i = i
        prec = x_diagonal[i] / (x_row_plus[i]+SMOOTH)
        print("\nForground of {}_precision= {:.3f}".format(i, prec))
        recall = x_diagonal[i] / (x_col_plus[i]+SMOOTH)
        print("{}_recall= {:.3f}".format(i, recall))
        F1_score = (2*recall*prec)/(recall+prec)
        print("{}_F1_score={:.3f}".format(i, F1_score))
        iou = x_diagonal[i] / (x_row_plus[i] + x_col_plus[i] - x_diagonal[i]+SMOOTH)
        print("{}_iou {:.3f}".format(i, iou))