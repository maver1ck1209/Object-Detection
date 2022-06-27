import pandas as pd
import glob
import os
import json
import pandas as pd

def xml_to_yolo_bbox(bbox, w, h):
    # xmin, ymin, xmax, ymax
    x_center = ((bbox[2] + bbox[0]) / 2) / w
    y_center = ((bbox[3] + bbox[1]) / 2) / h
    width = (bbox[2] - bbox[0]) / w
    height = (bbox[3] - bbox[1]) / h
    return [x_center, y_center, width, height]

classes = []
output_dir = "labels_test2"
image_dir = "test"
os.mkdir(output_dir)

data = pd.read_csv(r'C:\Users\VyshakB\Desktop\fasterrcn\models-master\models\research\object_detection\images\train_labels.csv')
df=data
gb=df.groupby('filename')

result = []
c=0
for i in df['filename'].unique():
    h=gb.get_group(i)
    filename1 = i
    for j in range(len(h)):
        label = h['class'].values[j]
        if label not in classes or not classes:
            classes.append(label)
        index = classes.index(label)
        pil_bbox=[h['xmin'].values[j],h['ymin'].values[j],h['xmax'].values[j],h['ymax'].values[j]]
        width = h['width'].values[j]
        height = h['height'].values[j]
        yolo_bbox = xml_to_yolo_bbox(pil_bbox, width, height)
        bbox_string = " ".join([str(x) for x in yolo_bbox])
        result.append(f"{index} {bbox_string}")
    if result:
        with open(os.path.join(output_dir, f"{filename1.split('.')[0]}.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(result))
    result = []
with open('classes.txt', 'w', encoding='utf8') as f:
    f.write(json.dumps(classes))
