""" This is the skeleton code for main.py
You need to complete the required functions. You may create addition source files and use them by importing here.
"""

import os
import cv2
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description='Get mIOU of video sequences')
    parser.add_argument('-i', '--inp_path', type=str, default='input', required=True, \
                                                        help="Path for the input images folder")
    parser.add_argument('-o', '--out_path', type=str, default='result', required=True, \
                                                        help="Path for the predicted masks folder")
    parser.add_argument('-c', '--category', type=str, default='b', required=True, \
                                                        help="Scene category. One of baseline, illumination, jitter, dynamic scenes, ptz (b/i/j/m/p)")
    parser.add_argument('-e', '--eval_frames', type=str, default='eval_frames.txt', required=True, \
                                                        help="Path to the eval_frames.txt file")
    args = parser.parse_args()
    return args

### copied 
# def get_gaussian(diff_sq_sum,mu,sigma_sq):
#     p = (2*np.pi)**(mu.shape[2]/2)
#     exponent = -0.5*diff_sq_sum/sigma_sq
#     gaussians = np.exp(exponent)/(p*(sigma_sq**0.5))
#     return gaussians 

# def get_background_gaussians(w,sigma,T):
#     w_ratio = -1*w/sigma
#     sorted_ratio_idx = np.argsort(w_ratio,axis=2)
#     w_ratio.sort(axis=2)
#     ratio_cumsum = np.cumsum(-1*w_ratio,axis=2)
#     threshold_mask = (ratio_cumsum < T)
#     background_gaussian_mask = np.choose(np.rollaxis(sorted_ratio_idx,axis=2),np.rollaxis(threshold_mask,axis=2))
#     return np.rollaxis(background_gaussian_mask,axis=0,start=3) 

# def get_masks(background_gaussians_mask,diff_sq_sum,lambda_sq,sigma_sq):
#     update_mask = background_gaussians_mask*(diff_sq_sum/sigma_sq < lambda_sq*sigma_sq)
#     foreground_mask = ~np.any(update_mask,axis=2)
#     mask_img = np.array(foreground_mask*255,dtype=np.uint8)
#     replace_mask = np.repeat(foreground_mask[...,None],background_gaussians_mask.shape[2],axis=2)
#     return update_mask, replace_mask, mask_img 

# def update(gaussians,alpha,w,mu,sigma_sq,diff_sq_sum,update_mask,replace_mask):
#     replace_mask_extended = np.repeat(replace_mask[:,:,None,:],mu.shape[2],axis=2)
#     update_mask_extended = np.repeat(update_mask[:,:,None,:],mu.shape[2],axis=2)
#     w = (1-alpha)*w + alpha*update_mask
#     w[replace_mask] = 0.0001
#     rho = alpha*gaussians
#     rho_extended = np.repeat(rho[:,:,None,:],mu.shape[2],axis=2)
#     mu[update_mask_extended] = (1-rho_extended[update_mask_extended])*mu[update_mask_extended] + rho_extended[update_mask_extended]*np.repeat(frame[...,None],mu.shape[3],axis=3)[update_mask_extended]
#     mu[replace_mask_extended] = np.repeat(frame[...,None],mu.shape[3],axis=3)[replace_mask_extended]
#     sigma_sq[replace_mask] = 16
#     sigma_sq[update_mask] = (1-rho[update_mask])*sigma_sq[update_mask] + rho[update_mask]*diff_sq_sum[update_mask]
#     sigma = np.sqrt(sigma_sq)
#     return w, mu, sigma_sq, sigma 


def baseline_bgs(args):
    #TODO complete this function
    ## eval frames from 470 to 1700  
    # src_path = '/home/sidd_s/assignments/assign_data/COL780/COL780-A1-Data/baseline'  
    opy_path2 = args.out_path.replace('pred_mog2', 'pred_knn') 
    ipx_lst = sorted(os.listdir(args.inp_path)) 
    # print(args.eval_frames)
    with open(args.eval_frames , 'r') as myfile: 
        data = myfile.read().splitlines()[0].split() 
    # print(data) 
    # print(ipx_lst[0])
    # ipx_lst = ipx_lst[469:] ## [470 1700]  ## why hard coding 
    # print(int(data[0])-1)  
    ipx_lst = ipx_lst[int(data[0])-1: int(data[1])]     
    # print(len(ipx_lst)) 

    ## background substraction modules 
    # back_sub = cv2.createBackgroundSubtractorMOG2(detectShadows=True) ## no change...and even with detectshadows = False
    # back_sub = cv2.createBackgroundSubtractorMOG2(varThreshold=45, history=1000, detectShadows=False)  ## mIOU: 0.5843 with 50 and with 1000 history mIOU: mIOU: 0.5908 with detect shadows false results are better than detect shawdows true and thresholding  ## here two much hyperparam tuning req to get better results in general ((0.6050 maxx))

    # kernel = None ## for morphological operation ## 0.6050 ## sames as 3*3 kernel results 
    # kernel = np.ones((3,3), np.uint8) ## 0.6050 ## morphological opening to reduce noise 

    # back_sub = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=50, detectShadows=True) ## decreasing
    # back_sub = cv2.createBackgroundSubtractorKNN(detectShadows=False) ## 0.6361
    back_sub = cv2.createBackgroundSubtractorKNN(detectShadows=True) ## with thresholding...0.7200(max).. it is fairly straight forward and quite general 
    # back_sub = cv2.createBackgroundSubtractorKNN(500, 500, detectShadows=True) ## less with this config # mIOU: 0.6235
    # back_sub2 = cv2.createBackgroundSubtractorKNN(varThreshold=45) 
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))   
    # back_sub = cv2.createBackgroundSubtractorGMG() ## will not work since its under bgsegm contrib module   

    # for ipx in ipx_lst:
    for ipx in range(1701):
        frame = cv2.imread(os.path.join(args.inp_path, ipx))
        # print(im.shape) 
        ## methods to apply  
        predpath = ipx.replace('in', 'pred')  
        predpath = predpath.replace('.jpg', '.png')
        # print(predpath)  

        # frame = cv2.medianBlur(frame, 11)
        pred_mask = back_sub.apply(frame)  

        if ipx in ipx_lst:
            _, pred_mask = cv2.threshold(pred_mask, 250, 255, cv2.THRESH_BINARY) ## for geting rid of shadows if any..## worked with knn method...yahoo!!
            # pred_mask = cv2.erode(pred_mask, kernel, iterations=1 ) ## morphological operation to get rid of the white noise (pepper type)  
            # pred_mask = cv2.dilate(pred_mask, kernel, iterations=2) ## ...//...
            pred_mask = cv2.medianBlur(pred_mask, 11)  # mIOU: 0.6886 ...with median filtering is better with 5x5 # mIOU: 0.7200 with median filter of 11x11...its better than the morphological way of removing noise
            # _, pred_mask = cv2.threshold(pred_mask, 250, 255, cv2.THRESH_BINARY)
            # pred_mask2 = back_sub2.apply(frame, learningRate = 0.5) ## change in lr, not working
            # pred_mask2 = back_sub2.apply(frame) 
            # pred_mask1 = cv2.morphologyEx(back_sub, cv2.MORPH_OPEN, kernel)  #  not working   
            # contours,_ = cv2.findContours(pred_mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) ## detect contours in a frame ...so as to detect only the ones which are cars and not the superficial blobs or blocks (noise) which are left over by erosion and dilation

            # for cnt in contours:
            #     if cv2.contourArea(cnt)>400: ## to detect only cars 
            #         print(cnt)
                    # break
                    # pred_mask[cnt] = 255
        
            # mask1 = cv2.cvtColor(mask1, cv2.COLOR_BGR2GRAY)   
            # print(mask1.shape)
            # print(mask2.shape)   
            ## thresholding  
            # print(mask1.shape) # (240, 320)
            # print(mask2.shape) # (240, 320)
            # print(np.unique(pred_mask1))
            # print(np.unique(pred_mask2)) 
            # break  
            
            # cv2.imwrite(os.path.join(args.out_path, predpath), pred_mask)  ## mog2 ## mIoU: 0.6050
            cv2.imwrite(os.path.join(opy_path2, predpath), pred_mask)      ## knn ## mIOU: 0.7200
            # cv2.imwrite(predpath, pred_mask1) ## still getting 3 channel image .. its ok
            # print(np.unique(pred_mask1))
            # break  
            # print('done')

            # ## copied 
            # # Initial values of parameters
            # K = 3
            # lambda_sq = 2.5**2
            # alpha = 0.2
            # T = 0.7
            # w = np.full((frame.shape[0],frame.shape[1],K),1/K)
            # mu = np.zeros(frame.shape+tuple([K]))
            # sigma = np.ones(w.shape)
            # sigma_sq = sigma
            # diff = frame[...,None] - mu
            # diff_sq_sum = np.sum(diff*diff,axis=2) 
    
    print('yo') 
    return 


def illumination_bgs(args):
    #TODO complete this function
    ## eval frames from 470 to 1700  
    # src_path = '/home/sidd_s/assignments/assign_data/COL780/COL780-A1-Data/baseline'  
    opy_path2 = args.out_path.replace('pred_mog2', 'pred_knn') 
    ipx_lst = sorted(os.listdir(args.inp_path)) 
    # print(args.eval_frames)
    with open(args.eval_frames , 'r') as myfile: 
        data = myfile.read().splitlines()[0].split() 
    # print(data) 
    # print(ipx_lst[0])
    # ipx_lst = ipx_lst[469:] ## [470 1700]  ## why hard coding 
    # print(int(data[0])-1)  
    ipx_lst = ipx_lst[int(data[0])-1: int(data[1])]     
    # print(len(ipx_lst))  

    # kernel = None ## for morphological operation ## 0.6050 ## sames as 3*3 kernel results 
    kernel = np.ones((3,3), np.uint8) ## 0.6050 ## morphological opening to reduce noise 

    back_sub = cv2.createBackgroundSubtractorKNN(detectShadows=True)
    for ipx in ipx_lst:
        frame = cv2.imread(os.path.join(args.inp_path, ipx))
        # print(im.shape) 
        ## methods to apply  
        predpath = ipx.replace('in', 'pred')  
        predpath = predpath.replace('.jpg', '.png')
        # print(predpath)  

        # frame = cv2.medianBlur(frame, 11)
        pred_mask = back_sub.apply(frame)  
        _, pred_mask = cv2.threshold(pred_mask, 250, 255, cv2.THRESH_BINARY) ## for geting rid of shadows if any..## worked with knn method...yahoo!!
        pred_mask = cv2.erode(pred_mask, kernel, iterations=1) ## morphological operation to get rid of the white noise (pepper type)  
        pred_mask = cv2.dilate(pred_mask, kernel, iterations=2) ## ...//...  here moropho is performing better than median bluring ## 0.2315 (current max)
        # pred_mask = cv2.medianBlur(pred_mask, 3)  
        pred_mask =  cv2.resize(pred_mask, (320,240))
        cv2.imwrite(os.path.join(opy_path2, predpath), pred_mask) 

    print('yo')
    return 


def jitter_bgs(args):
    #TODO complete this function
    pass

def dynamic_bgs(args):
    #TODO complete this function
    pass

def ptz_bgs(args):
    #TODO: (Optional) complete this function
    pass


def main(args):
    if args.category not in "bijdp":
        raise ValueError("category should be one of b/i/j/m/p - Found: %s"%args.category)
    FUNCTION_MAPPER = {
            "b": baseline_bgs,
            "i": illumination_bgs,
            "j": jitter_bgs,
            "m": dynamic_bgs,
            "p": ptz_bgs
        }

    FUNCTION_MAPPER[args.category](args)

if __name__ == "__main__":
    args = parse_args()
    main(args)


# python main.py -i /home/sidd_s/assignments/assign_data/COL780/COL780-A1-Data/baseline/input -c b -e /home/sidd_s/assignments/assign_data/COL780/COL780-A1-Data/baseline/eval_frames.txt -o /home/sidd_s/assignments/assign_data/COL780/COL780-A1-Data/baseline/pred_mog2