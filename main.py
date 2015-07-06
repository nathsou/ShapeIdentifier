import cv2, numpy as np

cap = cv2.VideoCapture(0)
acceptedPolygons = range(3, 7)

thresh = [50, 80]
regPolygon = 15
thickness = 2
angle = lambda ((ax, ay), (bx, by)), ((cx, cy), (dx, dy)): ((np.arccos((((ax - bx) * (cx - dx)) + ((ay - by) * (cy - dy)))/(((ax - bx)**2 + (ay - by)**2)**0.5 * ((cx - dx)**2 + (cy - dy)**2)**0.5)))/np.pi)*180

def ChangeThresh1(v):
    thresh[0] = v

def ChangeThresh2(v):
    thresh[1] = v

def setLabel(im, text, contour):
    labelSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
    bx, by, bw, bh = cv2.boundingRect(contour)
    x, y = (bx + (bw - labelSize[0][0])/2, (by + (bh + labelSize[0][1])/2)-10)
    cv2.rectangle(im, (x, y), (x + labelSize[0][0], y - labelSize[0][1]), (255, 255, 255), -1)
    cv2.putText(im, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

def recogShapes(src, shapesOnly = False):
    if shapesOnly:
        im = np.zeros((src.shape[0], src.shape[1], 3), src.dtype)
        im[:] = (241, 240, 236)
    else:
        im = src
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, thresh[0], thresh[1])
    cv2.equalizeHist(edges, edges)
    cv2.blur(edges, (2, 2), edges)
    (cnts, _) = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 100:
            approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
            if cv2.isContourConvex(approx):
                if len(approx) in acceptedPolygons:
                    angles = []
                    sides = []
                    for i in range(len(approx)):
                        p = approx[(i - 1) % len(approx)][0]
                        c = approx[i][0]
                        n = approx[(i + 1) % len(approx)][0]
                        a = angle(((c[0], c[1]), (p[0], p[1])), ((c[0], c[1]), (n[0], n[1])))
                        angles.append(a)
                        sides.append(((c[0] - p[0])**2 + (c[1] - p[1])**2)**0.5)
                        shapeName = str(len(approx)) + '-agon'
                        if np.std(angles) < regPolygon and np.std(sides) < regPolygon: #It's a regular polygon
                            color = (185, 128, 41)
                        else:
                            color = (96, 174, 39)
                        cv2.drawContours(im, [cnt], -1, color, thickness)
                        if np.fabs(a - 90) < 2:
                            cv2.rectangle(im, (c[0] - 3, c[1] - 3), (c[0] + 3, c[1] + 3), (173, 68, 142), 2) #Right angle
                        else:
                            cv2.circle(im, (c[0], c[1]), 3, (173, 68, 142), 2)
                            cv2.putText(im, str(round(a, 1)), (c[0], c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                        setLabel(im, shapeName, cnt)
                else: #Circle or Ellipse
                    rect = cv2.minAreaRect(cnt)
                    cv2.ellipse(im, rect, (43, 57, 192), thickness)
                    shape = 'ellipse'
                    if np.fabs(1 - rect[1][0]/rect[1][1]) < 0.1:
                        shape = 'circle'
                    setLabel(im, shape, cnt)

    return im

while True:
    ret, im = cap.read()
    cv2.imshow('Shapes', recogShapes(im.copy()))
    cv2.createTrackbar('Thresh1', 'Shapes', thresh[0], 300, ChangeThresh1)
    cv2.createTrackbar('Thresh2', 'Shapes', thresh[1], 300, ChangeThresh2)
    if cv2.waitKey(17) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
