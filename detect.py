from ultralytics import YOLO

if __name__ == '__main__':

    # Load a model
    model = YOLO(model='E:/yolo12/yolov12-main/yolov12-main/final_n.pt')
    model.predict(source='E:/yolo12/yolov12-main/yolov12-main/vd.v4i.yolov12/test/images',
                  save=True,
                  show=False,
                  )