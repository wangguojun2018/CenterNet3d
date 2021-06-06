import numpy as np
from pathlib import Path
import argparse
from mmdet3d.utils.draw_tools import Object3d,Calibration,read_lidar,draw_lidar,draw_gt_boxes3d,\
    project_velo_to_image,draw_projected_boxes3d
from mmdet3d.core.bbox.structures import CameraInstance3DBoxes,LiDARInstance3DBoxes,Box3DMode
import cv2
import torch
import glob
import os
os.chdir('/home/wgj/source_code/py/mmdetection3d')

def get_objects_from_label(label_file):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects = [Object3d(line) for line in lines]
    return objects



pred="/home/wgj/dataset/kitti/object/testing/partA2Pedestrian"
gt="/home/wgj/dataset/kitti/object/training/label_lidar"
window_name="part"
parser=argparse.ArgumentParser()
parser.add_argument("--label_dir", type=str,default=pred)
parser.add_argument("--gt_dir",type=str,default=gt)
parser.add_argument("--cloud_dir",type=str,default="/home/wgj/dataset/kitti/object/testing/velodyne_reduced")
parser.add_argument("--image_dir",type=str,default="/home/wgj/dataset/kitti/object/testing/image_2")
parser.add_argument("--calib_dir",type=str,default="/home/wgj/dataset/kitti/object/testing/calib")
parser.add_argument("--add_gt",action="store_true",default=False)
parser.add_argument("--save",action="store_true",default=True)
parser.add_argument("--save_dir",type=str,default="/home/wgj/Pictures/boshi/")
parser.add_argument("--idx", type=int,default=1)
parser.add_argument("--draw_cloud",action="store_true",default=False)
parser.add_argument("--draw_image",action="store_true",default=True)
args=parser.parse_args()

Path(args.save_dir).mkdir(exist_ok=True,parents=True)
label_files=glob.glob(args.label_dir+'/*.txt')
label_files.sort()
# label_files=label_files[4000:]
for label in label_files:
    idx = Path(label).stem
    print("current id is ",idx)
    cloud_path = Path(args.cloud_dir) / (idx + ".bin")
    image_path = Path(args.image_dir) / (idx + ".png")
    calib_path = Path(args.calib_dir) / (idx + ".txt")
    label_path = Path(args.label_dir) / (idx + ".txt")
    gt_path=Path(args.gt_dir)/(idx+".txt")

    calib = Calibration(str(calib_path))
    objects = get_objects_from_label(str(label_path))
    if len(objects)>0:
        boxes = np.vstack([obj.box3d for obj in objects])
    else:
        continue

    rect = calib.R0.astype(np.float32)
    Trv2c = calib.V2C.astype(np.float32)
    rect_4x4 = np.zeros([4, 4], dtype=rect.dtype)
    rect_4x4[3, 3] = 1.
    rect_4x4[:3, :3] = rect
    Trv2c_4x4 = np.concatenate([Trv2c, np.array([[0., 0., 0., 1.]])], axis=0)

    # box_instances = CameraInstance3DBoxes(boxes).to(torch.device('cuda')).convert_to(
    #         Box3DMode.LIDAR, np.linalg.inv(rect_4x4 @ Trv2c_4x4))

    box_instances_lidar = LiDARInstance3DBoxes(boxes,origin=(0.5, 0.5, 0.5)).to(torch.device('cuda'))

    # box_instances=box_instances_lidar.enlarged_box(0.15)
    corners = box_instances_lidar.corners.cpu().numpy()

    if args.draw_cloud:

        cloud = read_lidar(str(cloud_path))
        #     z=np.ones(cloud.shape[0])
        #     xyz=torch.from_numpy(cloud[:,:3]).to(torch.device('cuda'))
        #     points_box_id = box_instances.points_in_boxes(xyz).cpu().numpy()
        #
        #     cloud_bg=cloud[points_box_id<0]
        #     cloud_fg=cloud[points_box_id>-1]
        #
        #     cloud_bg.tofile("000116_bg.bin")
        #     cloud_fg.tofile("000116_fg.bin")



        f = draw_lidar(cloud, scale=10,axis=False, show=False,color=(1,1,1),name=window_name)
            # f = draw_lidar(cloud_fg,f, scale=0.1,axis=False, show=True, color=(0, 1, 0))
        if args.add_gt:
            f = draw_gt_boxes3d(corners, f, draw_text=False, show=False)
            gt_objects=get_objects_from_label(gt_path)
            gt_boxes=np.vstack([obj.box3d for obj in gt_objects])
            box_instances_gt = LiDARInstance3DBoxes(gt_boxes, origin=(0.5, 0.5, 0.5)).to(torch.device('cuda'))
            gt_corners = box_instances_gt.corners.cpu().numpy()
            f = draw_gt_boxes3d(gt_corners, f, color=(1,0,0), draw_text=False, show=True)
        else:
            f = draw_gt_boxes3d(corners, f, draw_text=False, show=True)


    if args.draw_image:
        image = cv2.imread(str(image_path))

        corners2d = project_velo_to_image(corners.reshape(-1, 3), calib).reshape(-1, 8, 2)
        image = draw_projected_boxes3d(image, corners2d)
        image = cv2.resize(image, (1500, 500))

        # if args.add_gt:
        #     gt_corners2d=project_velo_to_image(gt_corners.reshape(-1,3),calib).reshape(-1,8,2)
        #     image=draw_projected_boxes3d(image,gt_corners2d,color=(255,0,0))
        if args.save:
            file_name = Path(args.save_dir) / (idx + "_proj.png")
            cv2.imwrite(str(file_name), image)
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # cv2.setWindowProperty('image', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.resizeWindow(window_name,1500,600)
        cv2.moveWindow(window_name, 100, 100)
        cv2.imshow(window_name, image)

        while True:
            if cv2.waitKey(1000)==27:
                cv2.destroyAllWindows()
                break

        cv2.waitKey(0)






