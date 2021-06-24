# python interpreter searchs these subdirectories for modules
import sys
sys.path.insert(0, './yolov5')
sys.path.insert(0, './sort')

import argparse
import os
import platform
import shutil
import time
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

#yolov5
from yolov5.utils.datasets import LoadImages, LoadStreams
from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.utils.torch_utils import select_device, time_synchronized
import lane.lane as lane
import lane.lane_roi as lane_roi

#SORT
import skimage
from sort import *

torch.set_printoptions(precision=3)

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def bbox_rel(*xyxy):
    """" Calculates the relative bounding box from absolute pixel values. """
    bbox_left = min([xyxy[0].item(), xyxy[2].item()])
    bbox_top = min([xyxy[1].item(), xyxy[3].item()])
    bbox_w = abs(xyxy[0].item() - xyxy[2].item())
    bbox_h = abs(xyxy[1].item() - xyxy[3].item())
    x_c = (bbox_left + bbox_w / 2)
    y_c = (bbox_top + bbox_h / 2)
    w = bbox_w
    h = bbox_h
    return x_c, y_c, w, h


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)

# def predict_risk()

def draw_boxes(region,dx,dy,ds,img, bbox, identities=None, categories=None, names=None, offset=(0, 0)):
    score = 0
    score1 = 0
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        cx = round((x1+x2)/2) # box center 좌표 
        cy = round((y1+y2)/2)
        cat = int(categories[i]) if categories is not None else 0
        id = int(identities[i]) if identities is not None else 0
        
        #color = compute_color_for_labels(id)
        label = f'{names[cat]} | {id}'
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
        
        w = x2 - x1
        h = y2 - y1
         
        # box 크기 변화율에 따라 근사화 하여 다음 box의 크기를 결정
        dx1 = round(int(ds[id-1])/(2*w + 1.6*h))
        dy1 = round(0.8*dx1)


        # 각 tracking id 마다 prediction한 cx,cy를 결정 
        predy = cy + int(dy[id-1])
        predx = cx + int(dx[id-1])

        # box 크기 변화율에 따른 미래 box의 (x1,y1), (x2,y2) 예측
        x1_ = int(predx-w/2)-dx1
        y1_ = int(predy-h/2)-dy1
        x2_ = int(predx+w/2)+dx1
        y2_ = int(predy+h/2)+dy1

        # box내부로 들어오는 위험지역을 검출하여 점수로 계산
        pred_score = np.count_nonzero(region[y1_:y2_,x1_:x2_,0])    
        real_score = np.count_nonzero(region[y1:y2, x1:x2,0])
        pred_risk_point = (pred_score/((x2_-x1_)*(y2_-y1_)+0.0000000001))*100
        real_risk_point = (real_score/((x2-x1)*(y2-y1)+0.0000000001))*100
        score1 += real_risk_point
        score += pred_risk_point

        # 위험지역에 들어온 box는 빨간색으로 표시하고, box 왼쪽위에 score 표시
        if x1_ < img.shape[1] and x2_ <img.shape[1] and y1_ < img.shape[0] and y2_ < img.shape[0]:
          if pred_risk_point == 0.0 and real_risk_point == 0.0:         
                cv2.rectangle(img, (x1, y1), (x2, y2),color, 1)
                cv2.rectangle(img, (x1_, y1_), (x2_, y2_),[0,255,0], 1)  
                cv2.putText(img, f'{pred_risk_point}', (x1_, y1_ +
                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 1)
          else:
                cv2.rectangle(img, (x1, y1), (x2, y2),[0,0,255], 1)
                cv2.rectangle(img, (x1_, y1_), (x2_, y2_),[0,0,255], 1) 
                cv2.putText(img, f'{pred_risk_point}', (x1_, y1_ +
                                  t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 0, 255], 1)

    # 점수 0~1로 정규화
    if score >= 50 :
        score = 1
    else:
        score = score/50
        
    if score1 >= 50 :
        score1 = 1
    else:
        score1 = score1/50
    return score, score1

def detect(opt, *args):
    st = 0
    out, source, weights, view_img, save_txt, imgsz, save_img, sort_max_age, sort_min_hits, sort_iou_thresh= \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.save_img, opt.sort_max_age, opt.sort_min_hits, opt.sort_iou_thresh
    
    webcam = source == '0' or source.startswith(
        'rtsp') or source.startswith('http') or source.endswith('.txt')
    # Initialize SORT
    sort_tracker = Sort(max_age=sort_max_age,
                       min_hits=sort_min_hits,
                       iou_threshold=sort_iou_thresh) # {plug into parser}

    # Directory and CUDA settings for yolov5
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load yolov5 model
    model = torch.load(weights, map_location=device)['model'].float() #load to FP32. yolov5s.pt file is a dictionary, so we retrieve the model by indexing its key
    model.to(device).eval()
    if half:
        model.half() #to FP16

    # Set DataLoader
    vid_path, vid_writer = None, None
    
    if webcam:
        view_img = True
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        dataset = LoadImages(source, img_size=imgsz)
    
    # get names of object categories from yolov5.pt model
    names = model.module.names if hasattr(model, 'module') else model.names 
    
    # Run inference
    t0 = time.time()
    img = torch.zeros((1,3,imgsz,imgsz), device=device) #init img
    
    # Run once (throwaway)
    _ = model(img.half() if half else img) if device.type != 'cpu' else None
    
    save_path = str(Path(out))
    txt_path = str(Path(out))+'/results.txt'

    ###################################################################
    # 필요한 변수 선언
    risk_txt_path = str(Path(out)) + '/results_risk.txt'
    risk_txt_path2 = str(Path(out)) + '/results_risk2.txt'
    point1 = 0
    point2 = 0
    pred_frame = 0
    u_d = torch.zeros((1000,15)).to(device)
    v_d = torch.zeros((1000,15)).to(device)
    s_d = torch.zeros((1000,15)).to(device)
    dx = torch.zeros((1000,1)).to(device)
    dy = torch.zeros((1000,1)).to(device)
    ds = torch.zeros((1000,1)).to(device)
    #####################################################################

    for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset): #for every frame

        ##########################################################
        # sum을 통해서 각 id 마다 15 프레임 만큼의 변화율을 더해줌
        dx = torch.sum(u_d,axis = 1)
        dy = torch.sum(v_d,axis = 1)
        ds = torch.sum(s_d,axis = 1)
        ##########################################################
        st = time.time()
        img= torch.from_numpy(img).to(device)
        img = img.half() if half else img.float() #unint8 to fp16 or fp32
        img /= 255.0 #normalize to between 0 and 1.
        if img.ndimension()==3:
            img = img.unsqueeze(0)
            
        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0] 

        # Apply NMS
        pred = non_max_suppression(
            pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred): #for each detection in this frame
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            ################################################
            # 위험지역 설정
            region = np.zeros((im0.shape))
            region = lane_roi.lane_roi(region)
            ##################################################

            s += f'{img.shape[2:]}' #print image size and detection report
            save_path = str(Path(out) / Path(p).name)

            # Rescale boxes from img_size (temporarily downscaled size) to im0 (native) size
            det[:, :4] = scale_coords(
                img.shape[2:], det[:, :4], im0.shape).round()
            
            for c in det[:, -1].unique(): #for each unique object category
                n = (det[:, -1] ==c).sum() #number of detections per class
                s += f' - {n} {names[int(c)]}'

            dets_to_sort = np.empty((0,6))

            # Pass detections to SORT
            # NOTE: We send in detected object class too
            for x1,y1,x2,y2,conf,detclass in det.cpu().detach().numpy():
                dets_to_sort = np.vstack((dets_to_sort, np.array([x1, y1, x2, y2, conf, detclass])))

            # Run SORT
            tracked_dets = sort_tracker.update(dets_to_sort)
            
            # draw boxes for visualization
            if len(tracked_dets)>0:
                bbox_xyxy = tracked_dets[:,:4]
                identities = tracked_dets[:, 8]
                categories = tracked_dets[:, 4]

                # box를 그리고 점수를 반환함
                point1, point2 = draw_boxes(region,dx,dy,ds,im0, bbox_xyxy, identities, categories, names)

            im0 = lane_roi.lane_roi(im0) # 정해진 위험지역을 그림 
            #im0 = lane_roi.lane(im0) # 차선인식하여 위험지역을 그림


            # Write detections to file. NOTE: Not MOT-compliant format.

            # tracking 알고리즘으로 부터 좌표 정보와 변화율 정보를 받아옴
            if save_txt and len(tracked_dets) != 0:
                for j, tracked_dets in enumerate(tracked_dets):
                    bbox_x1 = tracked_dets[0]
                    bbox_y1 = tracked_dets[1]
                    bbox_x2 = tracked_dets[2]
                    bbox_y2 = tracked_dets[3]
                    category = tracked_dets[4]
                    u_overdot = tracked_dets[5]
                    v_overdot = tracked_dets[6]
                    s_overdot = tracked_dets[7]
                    identity = tracked_dets[8]
                    #################################################################
                    #
                    u_d[int(identity)-1,pred_frame] = u_overdot
                    v_d[int(identity)-1,pred_frame] = v_overdot
                    s_d[int(identity)-1,pred_frame] = s_overdot
                    cx = round((bbox_x1+bbox_x2)/2)
                    cy = round((bbox_y1+bbox_y2)/2)
                    cx_pred = torch.round(cx + dx[int(identity)-1])
                    cy_pred = torch.round(cy + dy[int(identity)-1])
                    #################################################################
                    with open(txt_path, 'a') as f:
                        f.write(f'{frame_idx},{bbox_x1},{bbox_y1},{bbox_x2},{bbox_y2},{category},{u_overdot},{v_overdot},{s_overdot},{identity},\n')
            
                # txt 파일에 프레임별 위험도 점수 출력
                with open(risk_txt_path,'a') as f:
                    f.write(f'{point1}\n')
                with open(risk_txt_path2,'a') as f:
                    f.write(f'{point2}\n')

            # Stream image results(opencv)
            if view_img:
                cv2.imshow(p,im0)
                if cv2.waitKey(1)==ord('q'): #q to quit
                    raise StopIteration
            # Save video results
            if save_img:
                print('saving img!')
                if dataset.mode == 'images':
                    cv2.imwrite(save_path, im0)
                else:
                    print('saving video!')
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
                    vid_writer.write(im0)
            pred_frame += 1
            if pred_frame == 15:
                pred_frame = 0  

    if save_txt or save_img:
        # print('Results saved to %s' % os.getcwd() + os.sep + out)
        if platform == 'darwin':  # MacOS
            os.system('open ' + save_path)

    print('Done. (%.3fs)' % (time.time() - t0))
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str,
                        default='yolov5/weights/yolov5s.pt', help='model.pt path')
    # file/folder, 0 for webcam
    parser.add_argument('--source', type=str,
                        default='inference/images', help='source')
    parser.add_argument('--output', type=str, default='inference/output',
                        help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=1080,
                        help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float,
                        default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float,
                        default=0.4, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v',
                        help='output video codec (verify ffmpeg support)')
    parser.add_argument('--device', default='',
                        help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true',
                        help='display results')
    parser.add_argument('--save-img', action='store_true',
                        help='save video file to output folder (disable for speed)')
    parser.add_argument('--save-txt', action='store_true',
                        help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int,
                        default=[i for i in range(80)], help='filter by class') #80 classes in COCO dataset
    parser.add_argument('--agnostic-nms', action='store_true',
                        help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true',
                        help='augmented inference')
    
    #SORT params
    parser.add_argument('--sort-max-age', type=int, default=5,
                        help='keep track of object even if object is occluded or not detected in n frames')
    parser.add_argument('--sort-min-hits', type=int, default=2,
                        help='start tracking only after n number of objects detected')
    parser.add_argument('--sort-iou-thresh', type=float, default=0.2,
                        help='intersection-over-union threshold between two frames for association')
    args = parser.parse_args()
    args.img_size = check_img_size(args.img_size)
    print(args)
    with torch.no_grad():
        detect(args)
