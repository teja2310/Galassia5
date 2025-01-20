import time
from ultralytics import YOLO
start = time.time()


model = YOLO("./runs/detect/train25/weights/best.pt")

input_path = '../blindtest_output/images'

results = model(input_path)

for result in results:
    with open("../blindtest_output/results/" + result.path.split("\\")[-1].split(".")[0] +".txt", "w+") as f:
        for i, box in enumerate(result.boxes.xyxy.cpu().numpy()):
            f.write(str(0) + ' ' + str(box[0]) + " " + str(box[1]) + " " + str(box[2]) + " " + str(box[3]) + ' ' + str(result.boxes.conf[i].cpu().numpy()) + '\n')


end = time.time()

print('Inference time with gpu', end - start)