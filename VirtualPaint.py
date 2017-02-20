import numpy as np
import cv2
import math
from Gestures import *

pritisnuto=True
brojac=False
blue = (200,0,0)
green = (0,200,0)
red = (0,0,200)
black=(0,0,0)

dictionary={}
dictionary['red']=red
dictionary['green']=green
dictionary['blue']=blue
dictionary['black']=black
state='black'

last_gesture='NONE'
iscrtavanje=[]

radius_thresh=0.04 # factor of width of full frame
finger_thresh_l=2.0
finger_thresh_u=3.8
first_iteration=True
finger_ct_history=[0,0]
def mark_hand_center(frame_in,cont):
    max_d=0
    pt=(0,0)
    x,y,w,h = cv2.boundingRect(cont)
    for ind_y in xrange(int(y+0.3*h),int(y+0.8*h)): #around 0.25 to 0.6 region of height (Faster calculation with ok results) 0.3 0.8
        for ind_x in xrange(int(x+0.3*w),int(x+0.6*w)): #around 0.3 to 0.6 region of width (Faster calculation with ok results) 0.3 0.6
            dist= cv2.pointPolygonTest(cont,(ind_x,ind_y),True)
            if(dist>max_d):
                max_d=dist
                pt=(ind_x,ind_y)
    if(max_d>radius_thresh*frame_in.shape[1]):
        thresh_score=True
        cv2.circle(frame_in,pt,int(max_d),(255,0,0),2)
        #cv2.circle(frame_in, pt, int(max_d*1.2), (255, 0, 255), 2)
        cv2.circle(frame_in, pt, 4, (255, 255, 255), 2)
    else:
        thresh_score=False
    return frame_in,pt,max_d,thresh_score

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=(1,2))
    return np.argmin(dist_2)

def furthest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=(1,2))
    return np.argmax(dist_2)

def racunanje_maske_za_dlan(frame_in,pt,max_d,thresh,cnt):
    max_d=1.3*max_d
    cv2.circle(frame_in, pt, int(max_d), (255, 255, 0), 2)
    #x0,y0=pt
    y0, x0 = pt

    levi=(-1,-1)
    desni=(-1,-1)

    rast=-1

    linije=[]
    for angle in range(0,360,1):
        x = math.cos(angle * math.pi / 180) * max_d + x0
        y = math.sin(angle * math.pi / 180) * max_d + y0
        min_dist=closest_node([int(y),int(x)], cnt)
        pederano = np.asarray(cnt)
        #if thresh[int(y),int(x)]!=0:
        linije.append([int(y),int(x)])
        linije.append(pederano[min_dist][0])

        if angle!=0:
            distancione = np.sqrt((linije[angle][1] - linije[angle-1][1]) ** 2 + (linije[angle][0] - linije[angle-1][0]) ** 2)
            if distancione>rast:
                rast=distancione
                levi=linije[angle]
                desni=linije[angle-1]

    pts=np.asarray(linije)
    pts = pts.reshape((-1, 1, 2))
    pokusaj1 = np.zeros(thresh.shape, np.uint8)
    #cv2.polylines(pokusaj1, [pts], True, (255, 255, 255),thickness=0)
    cv2.fillPoly(pokusaj1, [pts], 255)
    cv2.imshow('pokusaj1',pokusaj1)

    pederaj=cv2.bitwise_and(thresh, thresh, mask=pokusaj1)
    cv2.imshow('pederaj',pederaj)

    pederaj=np.invert(pederaj)
    idemomatora=cv2.bitwise_and(thresh, thresh, mask=pederaj)
    cv2.imshow('idemomatora',idemomatora)

    image1, contours1, hierarchy1 = cv2.findContours(idemomatora, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # print "duzina ", len(contours)
    contours1 = sorted(contours1, key=lambda x: cv2.contourArea(x), reverse=True)

    drawings = np.zeros(temporary.shape, np.uint8)

    for contour in contours1:
        if len(contour)>50:
            hullz = cv2.convexHull(contour)
            cv2.drawContours(drawings, [contour], 0, (0, 255, 0), 0)
            cv2.drawContours(drawings, [hullz], 0, (0, 0, 255), 0)

            momentz = cv2.moments(contour)
            if momentz['m00'] != 0:
                cxx = int(momentz['m10'] / momentz['m00'])  # cx = M10/M00
                cyy = int(momentz['m01'] / momentz['m00'])  # cy = M01/M00
            centrr = (cxx, cyy)
            cv2.circle(drawings, centrr, 5, [255, 0, 0], 2)

            cv2.circle(drawings, pt, 5, [255,0,255],2)

            anglex = abs(math.atan2((pt[1] - centrr[1]), (pt[0] - centrr[0])) * 180 / math.pi)

            distancija = np.sqrt((pt[1] - centrr[1]) ** 2 + (pt[0] - centrr[0]) ** 2)
            #print distancija
            #print drawings.shape

            #cv2.line(drawings, (levi[0],levi[1]), (desni[0],desni[1]), [255, 255, 255], 2)
            #print drawings.shape[1]

            #print pt[1], pt[0]

            cv2.line(drawings, (0,pt[1]), (drawings.shape[1],pt[1]), [255, 255, 255], 2)

            #minimalna_distanca = closest_node([cyy, cxx], contour)

            #maksimalna_distanca = furthest_node([cyy, cxx], contour)

            #pederano = np.asarray(contour)
            #najbliza_tacka=pederano[minimalna_distanca][0]
            #najdalja_tacka = pederano[maksimalna_distanca][0]

            #cv2.line(drawings, (najdalja_tacka[0],najdalja_tacka[1]), (najbliza_tacka[0],najbliza_tacka[1]), [255, 255, 0], 2)



            #if anglex<180:
            if distancija<200:
                cv2.line(drawings, pt, centrr, [0,255,255], 2)

    cv2.imshow('drawings',drawings)


#racunanje uglova na osnovu pozicije prstiju
def calc_angles(finger_count,finger_pos,hand_center):
    angle=np.zeros(finger_count,dtype=int)
    for i in range(finger_count):
        y = finger_pos[i][1]
        x = finger_pos[i][0]
        angle[i]=abs(math.atan2((hand_center[1]-y),(x-hand_center[0]))*180/math.pi)
    return angle

def ugao(levi,srednji,desni):
    ba = levi - srednji
    bc = desni - srednji
    #print levi,srednji,desni
    #ba=np.subtract(levi,srednji)
    #bc=np.subtract(desni,srednji)

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    if np.degrees(angle)<90:
        print srednji[0],srednji[1]
        cv2.circle(img, (srednji[0],srednji[1]), 3, [0, 0, 255], 30)
        print np.degrees(angle)

def mark_fingers(frame_in, hull, pt, radius):
    global first_iteration
    global finger_ct_history
    finger = [(hull[0][0][0], hull[0][0][1])]

    j = 0

    cx = pt[0]
    cy = pt[1]

    for i in range(len(hull)):
        dist = np.sqrt((hull[-i][0][0] - hull[-i + 1][0][0]) ** 2 + (hull[-i][0][1] - hull[-i + 1][0][1]) ** 2)
        #cv2.circle(temporary, (hull[-i][0][0], hull[-i][0][1]), 3, [0, 0, 255], 30)
        #cv2.circle(temporary, (hull[-i + 1][0][0], hull[-i + 1][0][1]), 5, [0, 255, 0], 30)
        #ugao(hull[-i - 1][0], hull[-i][0], hull[-i + 1][0])
        if (dist > 25):
            if (j == 0):
                finger = [(hull[-i][0][0], hull[-i][0][1])]
            else:
                finger.append((hull[-i][0][0], hull[-i][0][1]))
            j = j + 1
            #cv2.circle(temporary, (hull[-i][0][0], hull[-i][0][1]), 3, [0, 0, 255], 30)
            #cv2.circle(temporary, (hull[-i + 1][0][0], hull[-i + 1][0][1]), 5, [0, 255, 0], 30)

            #cv2.putText(frame_in, str(j), (hull[-i][0][0] + 10, hull[-i][0][1] - 10),
                        #cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255), 1, 8)
            #cv2.putText(frame_in, str(j), (hull[-i + 1][0][0] + 10, hull[-i + 1][0][1] - 10),
                        #cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

    temp_len = len(finger)
    i = 0


    while (i < temp_len):
        dist = np.sqrt((finger[i][0] - cx) ** 2 + (finger[i][1] - cy) ** 2)
        if (dist < finger_thresh_l * radius or dist > finger_thresh_u * radius or finger[i][1] > cy + radius):
            finger.remove((finger[i][0], finger[i][1]))
            temp_len = temp_len - 1
        else:
            i = i + 1

    temp_len = len(finger)
    if (temp_len > 5):
        for i in range(1, temp_len + 1 - 5):
            finger.remove((finger[temp_len - i][0], finger[temp_len - i][1]))
    palm = [(cx, cy), radius]

    count_text = "FINGERS:" + str(temp_len)
    cv2.putText(frame_in, count_text, (int(0.62 * frame_in.shape[1]), int(0.88 * frame_in.shape[0])),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

    for k in range(len(finger)):
        cv2.circle(frame_in, finger[k], 10, 255, 2)
        cv2.line(frame_in, finger[k], (cx, cy), 255, 2)

    return frame_in, finger, palm

def find_gesture(frame_in,finger,palm):
    global state
    global last_gesture
    frame_gesture.set_palm(palm[0],palm[1])
    frame_gesture.set_finger_pos(finger)
    frame_gesture.calc_angles()
    gesture_found=DecideGesture(frame_gesture,GestureDictionary)
    gesture_text="GESTURE:"+str(gesture_found)
    cv2.putText(frame_in,gesture_text,(int(0.56*frame_in.shape[1]),int(0.97*frame_in.shape[0])),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,255),1,8)
    #ovde radi selekciju ako se nalazi u nivou kruga
    if(last_gesture=='DRAW' and gesture_found!='DRAW'):
        print 'USAOOO'
        if len(iscrtavanje)!=0:
            print iscrtavanje[-1]
            iscrtavanje.append((iscrtavanje[-1][0],False))
            print iscrtavanje[-1]
    last_gesture=gesture_found


    if(gesture_found=="V"):
        angle = calc_angles(2, finger, (cx, cy))
        ugao = abs(angle[0] - angle[1])
        if (ugao <= 20):
            x = (finger[1][0] + finger[1 - 1][0]) / 2
            y = (finger[1][1] + finger[1 - 1][1]) / 2
            print 'selektovo si ',str(x),'-',str(y)
            cv2.circle(frame_in, (x, y), 10, 255, 2)
            if(x>=0 and x<=90 and y>=0 and y<=50):
                state='red'
            elif(x>=100 and x<=190 and y>=0 and y<=50):
                state='green'
            elif(x>=200 and x<=290 and y>=0 and y<=50):
                state='blue'
    elif(gesture_found=='DRAW'):
        iscrtavanje.append(((palm[0],state),True))
    cv2.putText(frame_in, 'COLOR: '+state, (int(0.56 * frame_in.shape[1]), int(0.1 * frame_in.shape[0])),
                cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

    return frame_in,gesture_found

def farthest_point(defects, contour, centroid):
    s = defects[:, 0][:, 0]
    cx, cy = centroid

    x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
    y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

    xp = cv2.pow(cv2.subtract(x, cx), 2)
    yp = cv2.pow(cv2.subtract(y, cy), 2)
    dist = cv2.sqrt(cv2.add(xp, yp))

    dist_max_i = np.argmax(dist)

    if dist_max_i < len(s):
        farthest_defect = s[dist_max_i]
        farthest_point = tuple(contour[farthest_defect][0])
        return farthest_point
    else:
        return None

def pomocna(privremena, defects, hand_center, hand_radius):

    prsti_start = []
    prsti_far = []

    brojac=0
    iks=0
    ipsilon=0

    procena_centra=(0,0)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[-i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        procena_centra+=start+end+far
        iks+=start[0]+end[0]+far[0]
        ipsilon += start[1] + end[1] + far[1]
        #tuple(map(operator.add, procena_centra, start))
        #tuple(map(operator.add, procena_centra, end))
        #tuple(map(operator.add, procena_centra, far))

        s1, e1, f1, d1 = defects[-i + 1, 0]
        start1 = tuple(cnt[s1][0])
        end1 = tuple(cnt[e1][0])
        far1 = tuple(cnt[f1][0])
        brojac=i

        rastojanje = np.sqrt((start[0] - start1[0]) ** 2 + (start[1] - start1[1]) ** 2)
        if rastojanje > 18:
            prsti_start.append(start)
            prsti_far.append(far)


            # a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            # b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            # c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            # angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57

    print iks/(brojac*3),ipsilon/(brojac*3)

    cv2.circle(privremena, (int(iks/(brojac*3)),int(ipsilon/(brojac*3))), 10, [255, 255, 255], 3)


    prsti_far_len = len(prsti_far)
    prsti_start_len = len(prsti_start)

    #print prsti_far_len, prsti_start_len

    i = 0
    while (i < prsti_start_len):
        dist = np.sqrt((prsti_start[i][0] - hand_center[0]) ** 2 + (prsti_start[i][1] - hand_center[1]) ** 2)
        if (dist < 1.5 * hand_radius or dist > finger_thresh_u * hand_radius or prsti_start[i][1] > hand_center[1] + hand_radius/2):
        #if prsti_start[i][1] > hand_center[1]+hand_radius/2:
        #if dist < finger_thresh_l * hand_radius or dist > finger_thresh_u * hand_radius:
            prsti_start.remove((prsti_start[i][0], prsti_start[i][1]))
            prsti_start_len = prsti_start_len - 1
            prsti_far.pop(i)
            prsti_far_len = prsti_far_len - 1
        else:
            i = i + 1

    # if (prsti_start_len > 5):
    #    for i in range(1, prsti_start_len + 1 - 5):
    #        prsti_start.remove((prsti_start[prsti_start_len - i][0], prsti_start[prsti_start_len - i][1]))
    #        prsti_far.pop(prsti_far_len-i)

    for k in range(len(prsti_start)):
        cv2.circle(privremena, prsti_far[k], 5, [0, 255, 255], 5)
        cv2.circle(privremena, prsti_start[k], 5, [255, 0, 255], 5)
        cv2.line(privremena, prsti_far[k], prsti_start[k], [255, 255, 0], 2)
        cv2.line(privremena, hand_center, prsti_start[k], [255, 0, 0], 2)
        if k==len(prsti_start)-1:
            cv2.line(privremena,prsti_far[k],prsti_far[0],[255,255,255],2)
        elif k>=0 and k<len(prsti_start)-1:
            cv2.line(privremena,prsti_far[k],prsti_far[k+1],[255,255,255],2)


    cv2.circle(privremena, hand_center, int(hand_radius), (255, 0, 0), 2)
    cv2.circle(privremena, hand_center, 4, (255, 255, 255), 2)

    #print 'holi moli', prsti_start_len, prsti_far_len

    cv2.imshow('privremena', privremena)

def nothing(x): #needed for createTrackbar to work in python.
    pass

cap = cv2.VideoCapture(0)
cv2.namedWindow('temp', flags=cv2.WINDOW_NORMAL)
cv2.createTrackbar('Hue L', 'temp', 136, 255, nothing)
cv2.createTrackbar('Sat L', 'temp', 136, 255, nothing)
cv2.createTrackbar('Val L', 'temp', 21, 255, nothing)
cv2.createTrackbar('Hue H', 'temp', 179, 179, nothing)
cv2.createTrackbar('Sat H', 'temp', 255, 255, nothing)
cv2.createTrackbar('Val H', 'temp', 255, 255, nothing)
cv2.createTrackbar('Erode', 'temp', 0, 31, nothing)
cv2.createTrackbar('Dilate', 'temp', 0, 31, nothing)
cv2.createTrackbar('Blur', 'temp', 5, 255, nothing)

drawing=False
top_left_pt, bottom_right_pt = (-1,-1), (-1,-1)
GestureDictionary=DefineGestures()
frame_gesture=Gesture("frame_gesture")

while True:
        ret,img=cap.read()#Read from source
        img = cv2.flip(img, 1) #flipujem ga

        temporary=img.copy()
        privremena=img.copy()

        blur = cv2.getTrackbarPos('Blur', 'temp')
        #trebalo bi pretvoriti u gray prvo pa gausian blur pa dalje
        if(blur%2==0):
               blur=blur+1
        blury = cv2.GaussianBlur(img, (blur,blur), 0)
        hsv=cv2.cvtColor(blury,cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV",hsv)
        bl_temp=cv2.getTrackbarPos('Hue L', 'temp')
        gl_temp=cv2.getTrackbarPos('Sat L', 'temp')
        rl_temp=cv2.getTrackbarPos('Val L', 'temp')
        bh_temp=cv2.getTrackbarPos('Hue H', 'temp')
        gh_temp=cv2.getTrackbarPos('Sat H', 'temp')
        rh_temp=cv2.getTrackbarPos('Val H', 'temp')

        thresh=cv2.inRange(hsv,(bl_temp,gl_temp,rl_temp),(bh_temp,gh_temp,rh_temp))

        erode = cv2.getTrackbarPos('Erode', 'temp')
        if erode == 0:
                erode = 1
        kernel = np.ones((erode, erode), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2)

        dilate = cv2.getTrackbarPos('Dilate', 'temp')
        if dilate == 0:
                dilate = 1
        kernel = np.ones((dilate, dilate), np.uint8)
        thresh = cv2.erode(thresh, kernel, iterations=2)

        prdara=thresh.copy()
        cv2.imshow('prdara', prdara)

        res = cv2.bitwise_and(img, img, mask=thresh)

        global pritisnuto

        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        elif cv2.waitKey(1) & 0xFF == ord('r'):
                if pritisnuto:
                        pritisnuto=False
                        iscrtavanje=[]
                else:
                        pritisnuto=True
        elif cv2.waitKey(1) & 0xFF == ord('d'):
                iscrtavanje=[]

        if pritisnuto:
                cv2.rectangle(temporary, (0, 0), (0 + 90, 0 + 50), red, -1)
                cv2.rectangle(temporary, (100, 0), (100 + 90, 0 + 50), green, -1)
                cv2.rectangle(temporary, (200, 0), (200 + 90, 0 + 50), blue, -1)

                #res = cv2.cvtColor(res, cv2.COLOR_HSV2RGB)
                #res = cv2.cvtColor(res, cv2.COLOR_RGB2GRAY)
                #_, res = cv2.threshold(res, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                #image, contours, hierarchy = cv2.findContours(res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
                image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


                #print "duzina ", len(contours)
                contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
                if len(contours)!=0:
                    cnt = max(contours, key=lambda x: cv2.contourArea(x))

                    #print cnt
                    #pederano=np.asarray(cnt)
                    #print pederano

                    hull = cv2.convexHull(cnt)
                    moments = cv2.moments(cnt)
                    if moments['m00'] != 0:
                            cx = int(moments['m10'] / moments['m00'])  # cx = M10/M00
                            cy = int(moments['m01'] / moments['m00'])  # cy = M01/M00
                    centr = (cx, cy)
                    cv2.circle(temporary, centr, 5, [0, 0, 255], 2)


                    #racunanje ugla
                    m11=moments['m11']
                    m20 = moments['m20']
                    m02 = moments['m02']
                    #print m11,m20,m02

                    diff=m20-m02
                    if(diff==0):
                        if m11==0:
                            print '0'
                        elif m11>0:
                            print '45'
                        elif m11<0:
                            print '-45'
                    else:
                        theta = 0.5 * math.atan2(2 * m11, diff)
                        angle = np.degrees(theta)
                        print diff, m11

                        if ((diff > 0) and (m11 == 0)):
                            print '0'
                        elif ((diff < 0) and (m11 == 0)):
                            print '-90'
                        elif ((diff > 0) and (m11 > 0)):
                            print angle
                        elif ((diff > 0) and (m11 < 0)):
                            print 180+angle
                        elif ((diff < 0) and (m11 > 0)):
                            print angle
                        elif ((diff < 0) and (m11 < 0)):
                            print 180+angle

                    #endy = cy+40 * math.sin(math.radians(int(angle)))
                    #endx = cx+40 * math.cos(math.radians(int(angle)))
                    #cv2.line(temporary, centr, (int(endx), int(endy)), 255, 2)



                    #crvena kockica
                    x, y, w, h = cv2.boundingRect(cnt)
                    cv2.rectangle(temporary, (x, y), (x + w, y + h), (0, 0, 255), 0)

                    #svetlopravi ram
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(temporary, [box], 0, (0, 255, 255), 2)

                    #zuti krug
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    center = (int(x), int(y))
                    radius = int(radius)
                    cv2.circle(temporary, center, radius, (255, 255, 0), 2)

                    #elipsa
                    #ellipse = cv2.fitEllipse(cnt)
                    #cv2.ellipse(temporary, ellipse, (255, 0, 255), 2)

                    drawing = np.zeros(temporary.shape, np.uint8)
                    cv2.drawContours(drawing, [cnt], 0, (0, 255, 0), 0)
                    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)
                    cv2.drawContours(temporary, [cnt], 0, (0, 255, 0), 0)
                    cv2.drawContours(temporary, [hull], 0, (255, 0, 0), 0)

                    temporary, hand_center, hand_radius, hand_size_score = mark_hand_center(temporary, cnt)

                    if (hand_size_score):
                        racunanje_maske_za_dlan(temporary, hand_center, hand_radius, prdara, cnt)
                        temporary, finger, palm = mark_fingers(temporary, hull, hand_center, hand_radius)
                        temporary, gesture_found = find_gesture(temporary, finger, palm)

                    temp_centar = (((-1,-1),'black'),True)
                    for broj, centar in enumerate(iscrtavanje):
                        cv2.circle(temporary, centar[0][0], 3, dictionary[centar[0][1]], -1)
                        if broj != 0:
                            if temp_centar[1]:
                                cv2.line(temporary, temp_centar[0][0], centar[0][0], dictionary[centar[0][1]], 3)
                        temp_centar = centar
                    cv2.imshow('drawing', drawing)
                    cv2.imshow('temporary',temporary)

                    #conhull = cv2.convexHull(cnt, returnPoints=False)
                    #defects = cv2.convexityDefects(cnt, conhull)
                    #pomocna(privremena, defects, hand_center, hand_radius)


        cv2.imshow('Video', img)
        cv2.imshow('thresh', thresh)
        cv2.imshow('Res', res)





cap.release()
cv2.destroyAllWindows()


"""
# hull = cv2.convexHull(cnt, returnPoints=False)
# defects = cv2.convexityDefects(cnt, hull)

# count_defects = 0

# najgornji = farthest_point(defects=defects, contour=cnt, centroid=centr)
# iscrtavanje.append(najgornji)
# iscrtavanje.append(centr)
# cv2.circle(temporary, najgornji, 5, [0, 255, 255], 10)

# temporary, hand_center, hand_radius, hand_size_score = mark_hand_center(temporary, cnt)
# if (hand_size_score):
#    temporary, finger, palm = mark_fingers(temporary, hull, hand_center, hand_radius)

for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 57
        if angle <= 90:
                if far[1] < centr[1]:
                        count_defects += 1
                        #cv2.circle(temporary, far, 5, [255, 0, 255], -1)
        cv2.line(temporary, start, end, [255, 255, 255], 2)

        if start[1]<centr[1] & far[1]>start[1] & far[1]>end[1]:
            print "usaoooooooooooooooooooooooooooooo"
            #cv2.circle(temporary, far, 5, [0, 0, 255], 30)
            cv2.circle(temporary, start, 5, [0, 255, 0], 30)
            cv2.line(temporary, centr, start, [0, 255, 0], 2)
            #cv2.circle(temporary, end, 5, [255, 0, 0], 30)




temp_centar = (0, 0)
for broj, centar in enumerate(iscrtavanje):
        cv2.circle(temporary, centar, 3, (255, 255, 255), -1)
        if broj != 0:
                cv2.line(temporary, temp_centar, centar, (0, 0, 0), 3)
        temp_centar = centar


cv2.putText(img, "BROJ KONTURA - " + str(count_defects),
            (int(0.50 * img.shape[1]), int(0.08 * img.shape[0])),
            cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1, 8)

#if (len(cnt) > 100):
"""