# LesionDetection.msg
# YOLO detection result for a single frame

float32 x              # pixel x (center of bounding box)
float32 y              # pixel y (center of bounding box)
float32 confidence     # YOLO confidence score [0,1]
int32 frame_id         # source frame index
string class_name      # "benign" or "malignant"
float32 box_x1         # bounding box xyxy
float32 box_y1
float32 box_x2
float32 box_y2
