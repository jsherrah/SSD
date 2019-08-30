import torch.utils.data
import numpy as np
from ssd.structures.container import Container
import json
from PIL import Image
import bbutils as bb

class FTSYGrandDataset(torch.utils.data.Dataset):
    class_names = ('__background__',
                   'rightFoot', 'leftFoot')

    LABEL_RIGHT_FOOT = 0
    LABEL_LEFT_FOOT  = 1

    # split is train or test
    def __init__(self, data_dir, ann_file, split, transform=None, target_transform=None):
        self.data_dir = data_dir
        self.ann_file = ann_file
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        # Load the session list
        sessionsListFile = '%s/%s.txt' % (self.data_dir, self.split)
        with open(sessionsListFile, 'r') as f:
            self.sessionNames = f.read().splitlines()
        print('FTSYGrandDataset {} set: loading {} sessions:\n{}'.format(
            self.split, len(self.sessionNames), self.sessionNames
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
                    'rightFoot': v['rightFoot']['boundingBox'],
                    'leftFoot':  v['leftFoot']['boundingBox'],
                    }

        print('Loaded {} images from {} sessions'.format(len(self.imageFilenames), len(self.sessionNames)))
        #!!print(json.dumps(self.groundTruth, indent=4))
        # Shuffle the filenames.
        np.random.shuffle(self.imageFilenames)

    def __getitem__(self, index):
        fn = self.imageFilenames[index]

        # load the image as a PIL Image
        image = self.readImage(fn)

        # Get ground truth
        gt = self.groundTruth[fn]

        # load the bounding boxes in x1, y1, x2, y2 order.
        boxes = np.vstack([gt['rightFoot'], gt['leftFoot']]).astype(np.float32)
        # and labels
        labels = np.array([self.LABEL_RIGHT_FOOT, self.LABEL_LEFT_FOOT], dtype=np.int64)

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

    def readAnnotationsFile(self, fn):
        with open(fn, 'r') as f:
            dat = json.load(f)
        return dat

    def readImage(self, fn):
        image = Image.open(fn)#.convert("RGB")
        image = np.array(image)
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
                          ann_file="boundingBoxes3D.json",
                          split='test')
    print(ds)

    # Show some results
    import matplotlib.pyplot as plt
    plt.ion()
    plt.figure()

    for img, gt, i in ds:
        plt.clf()
        plt.imshow(img)
        bb.bbPlot(plt.gca(), gt['boxes'][0,:], colour=(1,0,0), thickness=2, filled=False)
        bb.bbPlot(plt.gca(), gt['boxes'][1,:], colour=(0,1,0), thickness=2, filled=False)
        plt.waitforbuttonpress()
