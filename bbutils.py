import cv2
import matplotlib.patches

# Bounding box utilities.

# Given bounds [left, right, top, bottom] returns [ulx, uly, w, h]
def bbFromBounds(bbLRTB):
    l, r, t, b = bbLRTB
    return [l, t, r-l, b-t]

def boundsFromBB(bb):
    l, r, t, b = bb[0], bb[0]+bb[2], bb[1], bb[1]+bb[3]
    return l, r, t, b

def bbIntersection(bb1, bb2):
    return bbFromBounds([max(bb1[0], bb2[0]), min(bb1[0] + bb1[2], bb2[0] + bb2[2]),
                         max(bb1[1], bb2[1]), min(bb1[1] + bb1[3], bb2[1] + bb2[3])])

def bbDraw(img, bb, colour=(0,255,0), thickness=1, filled=False):
    if filled:
        thickness = cv2.FILLED
    cv2.rectangle(img, tuple(bb[:2]), tuple(np.array(bb[:2]) + bb[2:4]), colour, thickness)
    return img

def bbCentre(bb):
    return [bb[0] + 0.5*bb[2], bb[1] + 0.5*bb[3]]

def bbPlot(ax, bb, colour=(0,1,0), thickness=1, filled=False):
    r = matplotlib.patches.Rectangle(bb[:2], bb[2], bb[3], color=colour, linewidth=thickness, fill=filled)
    ax.add_patch(r)
