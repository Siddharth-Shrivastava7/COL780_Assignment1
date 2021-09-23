import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-p', '--pred_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-g', '--gt_path', type=str, default='groundtruth', required=True, \
                                                        help="Path for the ground truth masks folder")
    args = parser.parse_args()
    return args


def binary_mask_iou(mask1, mask2):
    mask1_area = np.count_nonzero(mask1 == 255)
    mask2_area = np.count_nonzero(mask2 == 255)
    intersection = np.count_nonzero(np.logical_and(mask1==255,  mask2==255))
    union = mask1_area+mask2_area-intersection 
    # print(union)
    # print(intersection)
    if union == 0: 
        # only happens if both masks are background with all zero values
        iou = 0
    else:
        iou = intersection/union  
    return iou  


def main(args):
    # Note: make sure to only generate masks for the evaluation frames mentioned in eval_frames.txt
    # Keep only the masks for eval frames in <pred_path> and not the background (all zero) frames.
    filenames = sorted(os.listdir(args.pred_path)) 
    # print(filenames)
    ious = []
    for filename in filenames: 
        # print(filename) 
        pred_mask = cv2.imread(os.path.join(args.pred_path, filename))  
        # print(pred_mask.shape) 
        # pred_mask[pred_mask>=127] = 255 
        # pred_mask[pred_mask<127] = 0
        # print(np.unique(pred_mask)) 
        filename_gt = filename.replace('pred', 'gt') 
        # print(args.gt_path) 
        # print(filename_gt)
        gt_mask = cv2.imread(os.path.join(args.gt_path, filename_gt)) 
        # print(gt_mask.shape)
        iou = binary_mask_iou(gt_mask, pred_mask)
        # print(iou)
        ious.append(iou)
    print("mIOU: %.4f"%(sum(ious)/len(ious)))
    

if __name__ == "__main__":
    args = parse_args()
    main(args)