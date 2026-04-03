from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

# load model
model = YOLO("parts_seg_m_90ep_best.pt")

# predict
results = model.predict(
    source="i1.jpg",
    conf=0.5,
    iou=0.5,
    save=False
)

# نجيب الصورة بالـ drawing
img = results[0].plot()

# نحول BGR → RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# display
plt.imshow(img)
plt.axis("off")
plt.show()