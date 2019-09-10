import torch.utils.data
import numpy as np
from ssd.structures.container import Container
import json
from PIL import Image
import bbutils
import os

def inRange(x, lo, hi):
    return np.logical_and(lo <= x, x <= hi)

def clamp(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

def bbToCorners(bb):
    x, y, w, h = bb
    return [x, y, x+w, y+h]

def bbFromCorners(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2-x1, y2-y1]

def bbScaleToImg(img, bbIn):
    H, W = img.shape[:2]
    bbOut = [bbIn[0]*W, bbIn[1]*H, bbIn[2]*W, bbIn[3]*H]
    return bbOut

class FTSYGrandDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'rightFoot', 'leftFoot')

    # Class names start at 1, because in voc.py background is first class.
    LABEL_RIGHT_FOOT = 1
    LABEL_LEFT_FOOT  = 2

    # split is train or test
    def __init__(self, data_dir, sessionListFile, ann_file, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.sessionListFile = sessionListFile
        self.transform = transform
        self.target_transform = target_transform
        # this does not work for some reason.  multi-threading?
        # self.cachedInfo = {}

        # Load the session list
        sessionListFile = '%s/%s' % (self.data_dir, self.sessionListFile)
        with open(sessionListFile, 'r') as f:
            self.sessionNames = f.read().splitlines()
        print('FTSYGrandDataset {} set: loading {} sessions:\n'.format(
            os.path.splitext(sessionListFile)[0], len(self.sessionNames),
        ))
        # Compile a list of image filenames and corresponding ground truth.  Then shuffle them.
        self.imageFilenames = []
        # map from filename to ground truth
        self.groundTruth = {}
        for sesh in self.sessionNames:
            # Make full path
            seshPath = '{}/{}'.format(self.data_dir, sesh)
            # Annotations file
            annoFile = '{}/{}'.format(seshPath, self.ann_file)
            anno = self.readAnnotationsFile(annoFile)
            # The bit we want looks like this:
            #
            #     "boundingBoxes2D": {
            #          "/data/jamie/ftsy/grand/demoSessions/session_fee52d4322484e50b8370664f8bdbbf6/results/images/00000015.jpg": {
            #              "rightFoot": {
            #                  "boundingBox": [
            #                      11.293305277984132,
            #                      838.0508336153313,
            #                      733.1308362471317,
            #                      446.31384196803526
            #                  ],
            #
            annoBB = anno['boundingBoxes2D']
            for k, v in annoBB.items():
                self.imageFilenames.append(k)
                self.groundTruth[k] = {
                    'rightFoot': bbToCorners(v['rightFoot']['boundingBox']),
                    'leftFoot':  bbToCorners(v['leftFoot'] ['boundingBox']),
                    }

        print('Loaded {} images from {} sessions'.format(len(self.imageFilenames), len(self.sessionNames)))
        #!!print(json.dumps(self.groundTruth, indent=4))
        # Shuffle the filenames.
        np.random.shuffle(self.imageFilenames)

    def __getitem__(self, index):
        fn = self.imageFilenames[index]

        # load the image as a PIL Image
        image = self.readImage(fn)
        imgW, imgH = image.shape[1], image.shape[0]

        # Cache size for later in evaluation...
#        if fn not in self.cachedInfo:
#            imgInfo = {'width': imgW, 'height': imgH}
#            #print('caching info for fn = {}'.format(fn))
#            self.cachedInfo[fn] = imgInfo

        boxes, labels = self._get_annotation(fn)

        # Clamp to image dimensions.
        boxes[:,0] = clamp(boxes[:,0], 0, imgW-1)
        boxes[:,1] = clamp(boxes[:,1], 0, imgH-1)
        boxes[:,2] = clamp(boxes[:,2], 0, imgW-1)
        boxes[:,3] = clamp(boxes[:,3], 0, imgH-1)

        for i in range(boxes.shape[0]):
            bb = boxes[i,:]
            assert np.all(inRange(bb[0], 0, imgW-1)), 'corners= {}, img size= {}x{}'.format(bb, imgW, imgH)
            assert np.all(inRange(bb[2], 0, imgW-1)), 'corners= {}, img size= {}x{}'.format(bb, imgW, imgH)
            assert np.all(inRange(bb[1], 0, imgH-1)), 'corners= {}, img size= {}x{}'.format(bb, imgW, imgH)
            assert np.all(inRange(bb[3], 0, imgH-1)), 'corners= {}, img size= {}x{}'.format(bb, imgW, imgH)

        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        targets = Container(
            boxes=boxes,
            labels=labels,
        )
        # return the image, the targets and the index in your dataset
        return image, targets, index

    def __len__(self):
        return len(self.imageFilenames)

    def get_img_info(self, index):
        fn = self.imageFilenames[index]
        #if fn in self.cachedInfo:
        #    return self.cachedInfo[fn]
        #else:
        #print('get_img_info: loading image {}!\ncache={}'.format(fn, self.cachedInfo))
        image = self.readImage(fn)
        imgW, imgH = float(image.shape[1]), float(image.shape[0])
        imgInfo = {'width': imgW, 'height': imgH}
        #self.cachedInfo[fn] = dict(imgInfo)
        return imgInfo

    def get_annotation(self, index):
        fn = self.imageFilenames[index]
        return fn, self._get_annotation(fn)

    def _get_annotation(self, imageFn):
        # Get ground truth
        gt = self.groundTruth[imageFn]

        # load the bounding boxes in x1, y1, x2, y2 order.
        boxes = np.vstack([gt['rightFoot'], gt['leftFoot']]).astype(np.float32)

        if 0:  #!!
            # Make these relative to image dimensions.
            image = self.readImage(imageFn) #!!!
            imgW, imgH = float(image.shape[1]), float(image.shape[0])

            if 1:
                boxes /= [imgW, imgH, imgW, imgH]
                if 1:
                    # Clamp to image.  Yes, the can go outside.
                    boxes = np.maximum(boxes, 0.0)
                    boxes = np.minimum(boxes, 1.0)

                #print('boxes = {}'.format(boxes))
                for i in range(boxes.shape[0]):
                    bb = boxes[i,:]
                    assert np.all(inRange(bb, 0, 1)), 'corners = {}, img size = {}x{}'.format(bb, imgW, imgH)

        # and labels
        labels = np.array([self.LABEL_RIGHT_FOOT, self.LABEL_LEFT_FOOT], dtype=np.int64)

        return boxes, labels


    def readAnnotationsFile(self, fn):
        with open(fn, 'r') as f:
            dat = json.load(f)
        return dat

    def readImage(self, fn):
        image = Image.open(fn).convert("RGB")
        image = np.array(image)#.astype(np.float32)
        #print('image type = {}'.format(image.dtype))
        return image

    def __str__(self):
        s = 'FTSYGrandDataset has {} images:\n'.format(len(self))
        for img, gt, i in self:
            s += '\t{:8d}: right = {}, left = {}, filename = {}\n'.format(i, gt['boxes'][0,:], gt['boxes'][1,:], self.imageFilenames[i])
            if i > 10:
                s += '\t...\n'
                break
        return s

if __name__ == '__main__':
    ds = FTSYGrandDataset(data_dir="/data/jamie/ftsy/grand/demoSessions",
                          sessionListFile="test.txt",
                          ann_file="boundingBoxes3D.json",
                          )
    print(ds)

    # Show some results
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()

    for img, gt, i in ds:
        plt.clf()
        plt.imshow(img)
        bbutils.bbPlot(plt.gca(), bbFromCorners(bbScaleToImg(img, gt['boxes'][0,:])), colour=(1,0,0), thickness=2, filled=False)
        bbutils.bbPlot(plt.gca(), bbFromCorners(bbScaleToImg(img, gt['boxes'][1,:])), colour=(0,1,0), thickness=2, filled=False)
        assert gt['labels'][0] == FTSYGrandDataset.LABEL_RIGHT_FOOT
        assert gt['labels'][1] == FTSYGrandDataset.LABEL_LEFT_FOOT
        plt.waitforbuttonpress()
