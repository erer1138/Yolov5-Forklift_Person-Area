import cv2
import torch
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')

cap = cv2.VideoCapture("Test Videos/test8.mp4")

target_classes = ['forklift', 'person']

forklift_last_position = None
motion_threshold = 3
forklift_moving = False 

# Polygon points
pts = []

# Function to draw polygon (ROI)
def draw_polygon(event, x, y, flags, param):
    global pts
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        pts.append([x, y])
    elif event == cv2.EVENT_RBUTTONDOWN:
        pts = []

# Function to check if a point is inside a polygon
def inside_polygon(point, polygon):
    return cv2.pointPolygonTest(polygon, (point[0], point[1]), False) >= 0

cv2.namedWindow('Video')
cv2.setMouseCallback('Video', draw_polygon)

def preprocess(img):
    height, width = img.shape[:2]
    ratio = height / width
    img = cv2.resize(img, (640, int(640 * ratio)), interpolation=cv2.INTER_CUBIC)
    new_height, new_width = img.shape[0] * 2, img.shape[1] * 2
    img = cv2.resize(img, (new_width, new_height))
    return img

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_detected = frame.copy()
    frame = preprocess(frame)
    results = model(frame)

    detections = []

    # Using pandas to get the detected objects' data
    for index, row in results.pandas().xyxy[0].iterrows():
        center_x = None
        center_y = None

        if row['name'] in target_classes:
            name = str(row['name'])
            x1 = int(row['xmin'])
            y1 = int(row['ymin'])
            x2 = int(row['xmax'])
            y2 = int(row['ymax'])

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            # Write name
            cv2.putText(frame, name, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            # Draw center
            cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)

            detections.append({'name': name, 'center_x': center_x, 'center_y': center_y})

    if len(pts) >= 4:
        frame_copy = frame.copy()
        cv2.fillPoly(frame_copy, np.array([pts]), (0, 0, 255))
        frame = cv2.addWeighted(frame_copy, 0.3, frame, 0.9, 0)

        for detection in detections:
            name = detection['name']
            center_x = detection['center_x']
            center_y = detection['center_y']

            if inside_polygon((center_x, center_y), np.array([pts])):
                if name == 'forklift':
                    if forklift_last_position is not None:
                        distance = np.sqrt((center_x - forklift_last_position[0])**2 + (center_y - forklift_last_position[1])**2)
                        forklift_last_position = (center_x, center_y)
                        if distance > motion_threshold:
                            forklift_moving = True
                        else:
                            forklift_moving = False
                    else:
                        forklift_last_position = (center_x, center_y)
                        forklift_moving = False

                    if forklift_moving:
                        for detection in detections:
                            if detection['name'] == 'person':
                                person_center_x = detection['center_x']
                                person_center_y = detection['center_y']
                                if inside_polygon((person_center_x, person_center_y), np.array([pts])):
                                    print("person detect")
                                    cv2.putText(frame, "Person in Area", (720, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                                    cv2.circle(frame, (person_center_x, person_center_y), 5, (0, 0, 255), -1)
                                    break

    cv2.imshow("Video", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()