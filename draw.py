import cv2
import numpy as np
import torch
from model import Net, CLASSES

model = Net()
model.load_state_dict(torch.load('models/model_scratch-40epochs.pt', map_location='cpu'))
model.eval()

drawing = False  # true if mouse is pressed


# mouse callback function
def paint_draw(event, x, y, flags, param):
    global ix, iy, drawing, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            cv2.line(canvas, (ix, iy), (x, y), (255, 255, 255), 5)
            ix = x
            iy = y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        cv2.line(canvas, (ix, iy), (x, y), (255, 255, 255), 5)
        ix = x
        iy = y
    return x, y


FONT_SIZE = 1
THICKNESS = 1
FONT_COLOR = (255, 255, 255)
CORRECT_COLOR = (255, 0, 255)
WRONG_COLOR = (255, 255, 0)

DISPLAY_SIZE = (200, 640, 3)

canvas = np.zeros((480, 640, 3), dtype=np.uint8)
cv2.namedWindow("Quick Draw")

display = np.zeros(DISPLAY_SIZE, dtype=np.uint8)
cv2.namedWindow("Display")

cv2.setMouseCallback('Quick Draw', paint_draw)

found = True
true_cls = 'pants'
attempts = 0

while (1):
    cv2.imshow('Quick Draw', 255-canvas)
    cv2.imshow('Display', 255-display)

    if found:
        true_cls = np.random.choice(CLASSES, 1)[0]
        while true_cls == 'pants':
            true_cls = np.random.choice(CLASSES, 1)[0]
        found = False

    cv2.putText(display, f'Draw {true_cls}!!', (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, THICKNESS, cv2.LINE_AA)

    k = cv2.waitKey(1) & 0xFF

    if k == ord(" "):
        display = np.zeros(DISPLAY_SIZE, dtype=np.uint8)
        cv2.putText(display, f'Draw {true_cls}!!', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, FONT_COLOR, THICKNESS, cv2.LINE_AA)

        canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        # Blur image
        median = cv2.medianBlur(canvas_gs, 9)
        gaussian = cv2.GaussianBlur(median, (5, 5), 0)
        # Otsu's thresholding
        _, thresh = cv2.threshold(gaussian, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, contour_gs, _ = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if len(contour_gs):
            contour = sorted(contour_gs, key=cv2.contourArea, reverse=True)[0]
            # Check if the largest contour satisfy the condition of minimum area
            if cv2.contourArea(contour) > 3000:
                x, y, w, h = cv2.boundingRect(contour)

                if y - 50 > 0 and y + h + 50 < canvas_gs.shape[0] and x - 50 > 0 and x + w + 50 < canvas_gs.shape[1]:
                    test_image = canvas_gs[y - 50:y + h + 50, x - 50:x + w + 50]
                else:
                    test_image = canvas_gs[y:y + h, x:x + w]

                test_image = cv2.resize(test_image, (28, 28))

                test_image = np.array(test_image, dtype=np.float32)[None, None, :, :]
                cv2.imshow("Canvas", cv2.resize(test_image[0, 0], (300, 300)))

                test_image = torch.from_numpy(test_image)
                output = model(test_image)
                predicted_class = torch.argsort(-1 * output[0])

                preds = [CLASSES[int(p.numpy())] for p in predicted_class[:3]]

                print(preds)

                if true_cls in preds:
                    found = True
                    attempts = 0
                    display = np.zeros(DISPLAY_SIZE, dtype=np.uint8)
                    cv2.putText(display, f'Oh I know it is a {true_cls}!', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, CORRECT_COLOR, THICKNESS, cv2.LINE_AA)
                else:
                    attempts += 1
                    cv2.putText(display, f'You are drawing {preds[0]}', (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, WRONG_COLOR, THICKNESS, cv2.LINE_AA)
                    if attempts >= 3:
                        found = True
                        attempts = 0
                        display = np.zeros(DISPLAY_SIZE, dtype=np.uint8)
            else:
                print("The object drawn is too small. Please draw a bigger one!")
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    if k == 27: # Escape KEY
        break
cv2.destroyAllWindows()