import time
import numpy as np
import cv2
import os
import torch
import torchvision
from random import randint
from imresize import imresize
import torchvision.transforms as T

def nms(boxes, probs=None, overlapThresh=0.3):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes are integers, convert them to floats -- this
    # is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and grab the indexes to sort
    # (in the case that no probabilities are provided, simply sort on the
    # bottom-left y-coordinate)
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = y2

    # if probabilities are provided, sort on them instead
    if probs is not None:
        idxs = probs

    # sort the indexes
    idxs = np.argsort(idxs)

    # keep looping while some indexes still remain in the indexes list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the index value
        # to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of the bounding
        # box and the smallest (x, y) coordinates for the end of the bounding
        # box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have overlap greater
        # than the provided overlap threshold
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked
    return boxes[pick].astype("int")


image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
def list_images(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue
            # determine the file extension of the current file
            ext = filename[filename.rfind("."):].lower()
            # check to see if the file is an image and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the image and yield it
                imagePath = os.path.join(rootDir, filename)
                yield imagePath


def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    # 如果高和宽为None则直接返回
    if width is None and height is None:
        return image
    # 检查宽是否是None
    if width is None:
        # 计算高度的比例并并按照比例计算宽度
        r = height / float(h)
        dim = (int(w * r), height)
    # 高为None
    else:
        # 计算宽度比例，并计算高度
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(image, dim, interpolation=inter)
    # return the resized image
    return resized


def cal_roi_mseloss(input_frames, recon_frames, detector):
    cnt = input_frames.shape[0]
    roi_cnt = 0
    roi_mseloss = 0
    res_frames = recon_frames - input_frames
    for i in range(cnt):
        img = np.array(torchvision.transforms.ToPILImage()(input_frames[i]))
        rects = detector.detect(img)
        if rects[1] is not None:
            rects = detector.detect(img)[1].astype(np.int32)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h, *_) in rects])
            pick = nms(rects, probs=None, overlapThresh=0.65)
            for (xA, yA, xB, yB) in pick:
                xA, yA, xB, yB = max(0, xA), max(0, yA), max(0, xB), max(0, yB)
                roi_mseloss += torch.mean((res_frames[i][:, yA:yB+1, xA:xB+1]).pow(2))
                roi_cnt += 1
    if roi_cnt == 0:
        return 0
    return roi_mseloss / roi_cnt


def cal_roi_ploss(input_frames, recon_frames, detector, ploss):
    cnt = input_frames.shape[0]
    roi_cnt = 0
    roi_mseloss = 0
    face_inputs, face_recons = [], []
    res_frames = recon_frames - input_frames
    for i in range(cnt):
        input_img = np.array(torchvision.transforms.ToPILImage()(input_frames[i]))
        recon_img = np.array(torchvision.transforms.ToPILImage()(recon_frames[i]))
        rects = detector.detect(input_img)
        if rects[1] is not None:
            rects = detector.detect(input_img)[1].astype(np.int32)
            rects = np.array([[x, y, x + w, y + h] for (x, y, w, h, *_) in rects])
            pick = nms(rects, probs=None, overlapThresh=0.65)
            for (xA, yA, xB, yB) in pick:
                xA, yA, xB, yB = max(0, xA), max(0, yA), max(0, xB), max(0, yB)
                face_input = imresize(input_img[yA:yB+1, xA:xB+1, :], output_shape=(112, 112))
                face_recon = imresize(recon_img[yA:yB+1, xA:xB+1, :], output_shape=(112, 112))
                face_inputs.append(torchvision.transforms.ToTensor()(face_input).unsqueeze(0))
                face_recons.append(torchvision.transforms.ToTensor()(face_recon).unsqueeze(0))
                roi_mseloss += torch.mean((res_frames[i][:, yA:yB+1, xA:xB+1]).pow(2))
                roi_cnt += 1
    if roi_cnt == 0:
        return -1
    roi_ploss = ploss(torch.cat(face_recons).cuda(), torch.cat(face_inputs).cuda()) / 10000
    roi_mseloss /= roi_cnt
    print('roi_ploss:', roi_ploss.item(), 'roi_mseloss:', roi_mseloss.item())
    return roi_ploss + roi_mseloss


def creat_weight_map(frames, detector):
    cnt = frames.shape[0]
    img_h, img_w = frames.shape[2], frames.shape[3]
    weight_map = torch.zeros(cnt, 1, img_h, img_w, dtype=torch.float32).cuda()
    for i in range(cnt):
        img = np.array(torchvision.transforms.ToPILImage()(frames[i]))
        rects = detector.detect(img)
        if rects[1] is not None:
            rects = detector.detect(img)[1].astype(np.int32)
        else:
            break
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h, *_) in rects])
        pick = nms(rects, probs=None, overlapThresh=0.65)
        for (xA, yA, xB, yB) in pick:
            xA, yA, xB, yB = max(0, xA), max(0, yA), max(0, xB), max(0, yB)
            weight_map[i][:, yA:yB+1, xA:xB+1] = 1
    return weight_map


def random_weight_map(hr):
    cnt = hr.shape[0]
    img_h, img_w = hr.shape[2], hr.shape[3]
    scale_factors = [9, 7, 5, 3, 2]
    weight_map = torch.zeros(cnt, 1, img_h, img_w, dtype=torch.float32).cuda()
    weighted_img = torch.zeros(cnt, 3, img_h, img_w, dtype=torch.float32).cuda()
    for i in range(cnt):
        idx = randint(0, 4)
        scale_factor = scale_factors[idx]
        score = 0.2 * idx
        degradation = T.Compose([
            T.Resize(256//scale_factor),
            # T.GaussianBlur(scale_factor * 2 - 1),
            T.Resize(256)
        ])
        lr = degradation(hr)
        xA, yA, xB, yB = randint(0, 127), randint(0, 127), randint(128, 255), randint(128, 255)
        # xA, yA, xB, yB = randint(32, 96), randint(32, 96), randint(160, 224), randint(160, 224)
        weight_map[i] = score
        weight_map[i][:, yA:yB+1, xA:xB+1] = 1
        weighted_img[i] = hr[i] * (weight_map[i] == 1) + lr[i] * (weight_map[i] != 1)
    return weight_map, weighted_img


# def creat_weight_map(frames, device='cpu'):
#     hog = cv2.HOGDescriptor()
#     hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
#     cnt = frames.shape[0]
#     img_w, img_h = frames.shape[2], frames.shape[3]
#     # weight_map = torch.ones(cnt, img_w, img_h, dtype=torch.uint8)
#     # for i in range(cnt):
#     #     img = np.array(torchvision.transforms.ToPILImage()(frames[i]))
#     #     (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
#     #     rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#     #     pick = nms(rects, probs=None, overlapThresh=0.65)
#     #     for (xA, yA, xB, yB) in pick:
#     #         weight_map[i][yA:yB+1, xA:xB+1] = 0
#     weight_map = torch.ones(img_w, img_h, dtype=torch.uint8, device=device)
#     for i in range(cnt):
#         img = np.array(torchvision.transforms.ToPILImage()(frames[i]))
#         (rects, weights) = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.05)
#         rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
#         pick = nms(rects, probs=None, overlapThresh=0.65)
#         for (xA, yA, xB, yB) in pick:
#             weight_map[yA:yB+1, xA:xB+1] = 10
#     return weight_map



if __name__ == "__main__":
    # img = cv2.imread('../data/vimeo_septuplet/sequences/00005/0007/im1.png')
    # img_t = torchvision.transforms.ToTensor()(img).unsqueeze(0)
    # weight_map = creat_weight_map(img_t)
    # img_t *= weight_map
    # cv2.imshow("Restruct img", np.array(torchvision.transforms.ToPILImage()(img_t[0])))
    # cv2.waitKey(0)
    # for i in range(img_t.shape[0]):
    #     img_t[i] *= weight_map[i]
    #     img1 = np.array(torchvision.transforms.ToPILImage()(img_t[i]))
    # construct the argument parse and parse the arguments
    # ap = argparse.ArgumentParser()
    # ap.add_argument("-i", "--images", default='test1', help="path to images directory")
    # args = vars(ap.parse_args())
    # 初始化 HOG 描述符/人物检测器
    # hog = cv2.HOGDescriptor()
    # hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    # faceDetector = cv2.CascadeClassifier(r'D:\Software\anaconda3\envs\pytorch\Lib\site-packages\cv2\data\haarcascade_profileface.xml')
    faceDetector = cv2.FaceDetectorYN_create(model='../yunet.onnx', config='', input_size=(1280, 704))
    # loop over the image paths
    for imagePath in list_images(r'E:\DeepLearning\ROIDVC\ROIDVCdemo\data\UVG\images\002'):
        # 加载图像并调整其大小以
        # （1）减少检测时间
        # （2）提高检测精度
        imagePath = r'E:\DeepLearning\ROIDVC\ROIDVCdemo\data\UVG\images\002\im034.png'
        t1 = time.time()
        image = cv2.imread(imagePath)
        # image = resize(image, width=min(400, image.shape[1]))
        orig = image.copy()
        # print(image)
        # detect people in the image
        # (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
        # rects = faceDetector.detectMultiScale(image)
        rects = faceDetector.detect(image)[1].astype(np.int32)
        # draw the original bounding boxes
        # print(rects)
        for (x, y, w, h, *_) in rects:
            cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # 使用相当大的重叠阈值对边界框应用非极大值抑制，以尝试保持仍然是人的重叠框
        # rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
        rects = np.array([[x, y, x + w, y + h] for (x, y, w, h, *_) in rects])
        pick = nms(rects, probs=None, overlapThresh=0.65)
        # draw the final bounding boxes
        weight_map = np.ones((image.shape[0], image.shape[1]), dtype=np.uint8) * 255
        for (xA, yA, xB, yB) in pick:
            print((xA, yA, xB, yB))
            cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)
            weight_map[yA:yB+1, xA:xB+1] = 0
        weight_map = np.expand_dims(weight_map, 2)
        # show some information on the number of bounding boxes
        filename = imagePath[imagePath.rfind("/") + 1:]
        print("[INFO] {}: {} original boxes, {} after suppression".format(
            filename, len(rects), len(pick)))
        # show the output images
        # cv2.imshow("Before NMS", orig)
        t2 = time.time()
        print('time: ', t2 - t1)
        cv2.imshow("Weight", weight_map)
        cv2.imshow("After NMS", image)
        cv2.waitKey(0)

