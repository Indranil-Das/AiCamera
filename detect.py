import cv2
from utils.hubconf import custom
from utils.plots import plot_one_box
import os
import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", type=str, required=True,
                help="path to video path")
ap.add_argument("-m", "--model", type=str, required=True,
                help="path to best.pt (YOLOv7) model")
ap.add_argument("-o", "--outdir", type=str, required=True,
                help="path to output/dir")
ap.add_argument("-f", "--frame", type=int, required=True,
                help="number that save frame after")

args = vars(ap.parse_args())
path_to_video = args['input']
path_to_model = args['model']
frames_dir = args["outdir"]
frame_no = args['frame']

os.makedirs(frames_dir, exist_ok=True)
count = 0
model = custom(path_or_model=path_to_model)
'''
def count1(results,img):
  model_values=[]
  aligns=im0.shape
  align_bottom=aligns[0]
  align_right=(aligns[1]/1.7 ) 

  for i, (k, v) in enumerate(results.items()):
    a=f"{k} = {v}"
    model_values.append(v)
    align_bottom=align_bottom-35                                                   
    cv2.putText(im0, str(a) ,(int(align_right),align_bottom), cv2.FONT_HERSHEY_SIMPLEX, 1,(45,255,255),1,cv2.LINE_AA)
'''    
cap = cv2.VideoCapture(path_to_video)
while True:
  success, img = cap.read()
  if not success:
    break
  bbox_list = []
  results = model(img)
  # Bounding Box
  box = results.pandas().xyxy[0]
  # print(box.index)
  class_list = box['class'].to_list()
  for i in box.index:
    xmin, ymin, xmax, ymax = int(box['xmin'][i]), int(box['ymin'][i]), int(box['xmax'][i]), int(box['ymax'][i])
    bbox_list.append([xmin, ymin, xmax, ymax])
    
  # on_mobile
  if class_list[0]==1:
    count += 1
    if (count>=frame_no and (count%frame_no)==0) or count==2:
      for bbox in bbox_list:
        plot_one_box(bbox, img, label='on_mobile', color=[0,255,0], line_thickness=2)
      cv2.imwrite(f'{frames_dir}/{len(os.listdir(frames_dir))}.jpg', img)
      print(f'[INFO] {len(os.listdir(frames_dir))}.jpg Frame Saved Successfully')
      if not count==2:
        count = 0
print('[INFO] Task Completed Successfully')

'''
counted_classes = count_objects(pred_bbox, by_class = False, allowed_classes=allowed_classes)
            # loop through dict and print
            for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
            image = utils.draw_bbox(frame, pred_bbox, FLAGS.info, counted_classes, allowed_classes=allowed_classes, read_plate=FLAGS.plate)
taco = [category_index.get(value) for index,value in enumerate(classes[0]) if scores[0,index] > 0.5]
print(len(taco))
'''            