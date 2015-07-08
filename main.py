import cv2, numpy as np, argparse

ap = argparse.ArgumentParser()
ap.add_argument("-s", "--source", required=False, help="Source image path")
ap.add_argument("-o", "--output", required=False, help="Destination image path")
ap.add_argument("-t", "--thickness", required=False, help="Trait thickness", type=int)
ap.add_argument("-a", "--showAngles", required=False, action='store_true')
ap.add_argument("-n", "--showShapesName", required=False, action='store_true')
ap.add_argument("-v", "--verbose", required=False, action='store_true', help="Displays detected shapes's vertices")
args = vars(ap.parse_args())

acceptedPolygons = range(3, 7)

shapeNames = ['point', 'line', 'tri', 'quad', 'penta', 'hexa', 'hepta', 'octa']

regPolygon = 10
thickness = -1
if not args['thickness'] is None:
    thickness = int(args['thickness'])
showNames = not args['showShapesName']
showAngles = args['showAngles']

angle = lambda ((ax, ay), (bx, by)), ((cx, cy), (dx, dy)): ((np.arccos((((ax - bx) * (cx - dx)) + ((ay - by) * (cy - dy)))/(((ax - bx)**2 + (ay - by)**2)**0.5 * ((cx - dx)**2 + (cy - dy)**2)**0.5)))/np.pi)*180

def autoCanny(src, sigma=0.33):
    med = np.median(src)
    thresh1 = int(max(0, (1.0 - sigma) * med))
    thresh2 = int(min(255, (1.0 + sigma) * med))
    return cv2.Canny(src, thresh1, thresh2)

def setLabel(im, text, contour):
    if showNames:
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
    edges = autoCanny(gray)
    #cv2.blur(edges, (2, 2), edges)
    if thickness is -1:
        method = cv2.RETR_TREE
    else:
        method = cv2.RETR_EXTERNAL
    (cnts, _) = cv2.findContours(edges, method, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for cnt in cnts:
        if cv2.contourArea(cnt) > 750:
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
                        if showAngles:
                            if np.fabs(a - 90) < 2:
                                cv2.rectangle(im, (c[0] - 3, c[1] - 3), (c[0] + 3, c[1] + 3), (173, 68, 142), 2) #Right angle
                            else:
                                cv2.circle(im, (c[0], c[1]), 3, (173, 68, 142), 2)
                                cv2.putText(im, str(round(a, 1)), (c[0], c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
                    if len(approx) > len(shapeNames):
                        shapeName = str(len(approx)) + '-agon'
                    else:
                        shapeName = shapeNames[len(approx) - 1]
                    if args['verbose']:
                        print shapeName + ' ' + str([(p[0][0], p[0][1]) for p in approx])
                    if np.std(angles) < regPolygon and np.std(sides) < regPolygon: #It's a regular polygon
                        color = (185, 128, 41)
                    else:
                        color = (96, 174, 39)
                    for i in range(len(approx)):
                        c = approx[i][0]
                        n = approx[(i + 1) % len(approx)][0]
                        if thickness is -1:
                            cv2.drawContours(im, [cnt], -1, color, thickness)
                        else:
                            cv2.line(im, (c[0], c[1]), (n[0], n[1]), color, thickness)
                    setLabel(im, shapeName, cnt)
                else: #Circle or Ellipse
                    rect = cv2.minAreaRect(cnt)
                    cv2.ellipse(im, rect, (43, 57, 192), thickness)
                    shape = 'ellipse'
                    if np.fabs(1 - rect[1][0]/rect[1][1]) < 0.15:
                        shape = 'circle'
                    setLabel(im, shape, cnt)

    return im

if args['source'] is None:
    cap = cv2.VideoCapture(-1)
    while True:
        ret, im = cap.read()
        cv2.imshow('Shapes', recogShapes(im))
        if cv2.waitKey(17) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
else:
    im = cv2.imread(args['source'])
    out = recogShapes(im)
    cv2.imshow('Shapes', out)
    if not args['output'] is None:
        cv2.imwrite(args['output'], out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
