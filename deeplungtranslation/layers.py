import numpy as np

import tensorflow as tf
from tensorflow import keras
import math


class PostRes2d(tf.keras.Model):
    """
    PostRes2d class represents a residual block in a 2D convolutional neural network.

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.

    Attributes:
        conv1 (tf.keras.layers.Conv2D): 2D convolutional layer with specified number of output filters and kernel size.
        bn1 (tf.keras.layers.BatchNormalization): Batch normalization layer.
        relu (tf.keras.layers.ReLU): ReLU activation layer.
        conv2 (tf.keras.layers.Conv2D): 2D convolutional layer with specified number of output filters and kernel size.
        bn2 (tf.keras.layers.BatchNormalization): Batch normalization layer.
        shortcut (tf.keras.Sequential or None): Sequential layer consisting of Conv2D and BatchNormalization layers
                                                for the shortcut connection. None if no shortcut connection is needed.

    Methods:
        forward: Performs forward propagation through the residual block.

    Examples:
        >>> model = PostRes2d(64, 128, stride=2)
        >>> x = tf.random.normal((1, 32, 32, 64))
        >>> output = model.forward(x)
    """
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes2d, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=n_out, kernel_size=3, strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv2D(n_out, kernel_size=3, strides=1, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1 or n_out != n_in:
            self.shortcut = tf.keras.Sequential(
                tf.keras.layers.Conv2D(n_out, kernel_size=1, strides=stride),
                tf.keras.layers.BatchNormalization())
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class PostRes(tf.keras.Model):
    """
    Class representing a PostRes block in a neural network.

    Args:
        n_in (int): Number of input channels.
        n_out (int): Number of output channels.
        stride (int, optional): Stride value for the convolutional layers. Defaults to 1.

    Attributes:
        conv1 (tf.keras.layers.Conv3D): Convolutional layer 1.
        bn1 (tf.keras.layers.BatchNormalization): Batch Normalization layer 1.
        relu (tf.keras.layers.ReLU): ReLU activation layer.
        conv2 (tf.keras.layers.Conv3D): Convolutional layer 2.
        bn2 (tf.keras.layers.BatchNormalization): Batch Normalization layer 2.
        shortcut (tf.keras.layers.Sequential or None): Shortcut connection.

    Methods:
        forward: Performs forward pass through the PostRes block.

    """
    def __init__(self, n_in, n_out, stride=1):
        super(PostRes, self).__init__()
        self.conv1 = tf.keras.layers.Conv3D(n_out, kernel_size=3, strides=stride, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
        self.conv2 = tf.keras.layers.Conv3D(n_out, kernel_size=3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        if stride != 1 or n_out != n_in:
            self.shortcut = tf.keras.layers.Sequential(
                tf.keras.layers.Conv3D(n_out, kernel_size=1, strides=stride),
                tf.keras.layers.BatchNormalization())
        else:
            self.shortcut = None

    def forward(self, x):
        residual = x
        if self.shortcut is not None:
            residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        return out


class Rec3(tf.keras.Model):
    """

    This class represents a Rec3 model in a TensorFlow Keras implementation.

    Parameters:
    - n0 (int): Number of filters in the first convolutional block.
    - n1 (int): Number of filters in the second convolutional block.
    - n2 (int): Number of filters in the third convolutional block.
    - n3 (int): Number of filters in the fourth convolutional block.
    - p (float, optional): Dropout probability. Default is 0.0.
    - integrate (bool, optional): Whether to integrate the output with the input. Default is True.

    Attributes:
    - block01 (tf.keras.Sequential): Sequential model for the first block in the Rec3 model.
    - block11 (tf.keras.Sequential): Sequential model for the second block in the Rec3 model.
    - block21 (tf.keras.Sequential): Sequential model for the third block in the Rec3 model.
    - block12 (tf.keras.Sequential): Sequential model for the fourth block in the Rec3 model.
    - block22 (tf.keras.Sequential): Sequential model for the fifth block in the Rec3 model.
    - block32 (tf.keras.Sequential): Sequential model for the sixth block in the Rec3 model.
    - block23 (tf.keras.Sequential): Sequential model for the seventh block in the Rec3 model.
    - block33 (tf.keras.Sequential): Sequential model for the eighth block in the Rec3 model.
    - relu (tf.keras.layers.ReLU): ReLU activation function.
    - p (float): Dropout probability.
    - integrate (bool): Integration flag.

    Methods:
    - forward(x0, x1, x2, x3): Forward pass of the Rec3 model.

    """
    def __init__(self, n0, n1, n2, n3, p=0.0, integrate=True):
        super(Rec3, self).__init__()

        self.block01 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(n1, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(n1, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.block11 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(filters=n1, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(filters=n1, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.block21 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(n1, kernel_size=2, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(n1, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.block12 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(n2, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(n2, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.block22 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(filters=n2, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(filters=n2, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.block32 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(n2, kernel_size=2, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(n2, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.block23 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(n3, kernel_size=3, strides=2, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(n3, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.block33 = tf.keras.layers.Sequential([
            tf.keras.layers.Conv3D(filters=n3, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv3D(filters=n3, kernel_size=3, padding='same'),
            tf.keras.layers.BatchNormalization()])

        self.relu = tf.keras.layers.ReLU()
        self.p = p
        self.integrate = integrate

    def forward(self, x0, x1, x2, x3):
        if self.p > 0 and self.training:
            bernoulli_probs = tf.fill((8,), 1.0 - self.p)
            coef = tf.cast(tf.random.categorical(tf.math.log([bernoulli_probs, 1 - bernoulli_probs]), 1),
                           dtype=tf.float32)
            out1 = coef[0] * self.block01(x0) + coef[1] * self.block11(x1) + coef[2] * self.block21(x2)
            out2 = coef[3] * self.block12(x1) + coef[4] * self.block22(x2) + coef[5] * self.block32(x3)
            out3 = coef[6] * self.block23(x2) + coef[7] * self.block33(x3)
        else:
            out1 = (1 - self.p) * (self.block01(x0) + self.block11(x1) + self.block21(x2))
            out2 = (1 - self.p) * (self.block12(x1) + self.block22(x2) + self.block32(x3))
            out3 = (1 - self.p) * (self.block23(x2) + self.block33(x3))

        if self.integrate:
            out1 += x1
            out2 += x2
            out3 += x3

        return x0, self.relu(out1), self.relu(out2), self.relu(out3)


def hard_mining(neg_output, neg_labels, num_hard):
    """
    Perform hard mining on negative output and labels.

    :param neg_output: A tensor representing the negative output.
    :param neg_labels: A tensor representing the negative labels.
    :param num_hard: The number of hard examples to retain.

    :return: A tuple containing the selected negative output and labels after hard mining.
    """
    _, idcs = tf.math.top_k(neg_output, min(num_hard, len(neg_output)))
    #make sure idcs is a tensor
    neg_output = tf.gather(neg_output, 0, idcs)
    neg_labels = tf.gather(neg_labels, 0, idcs)
    return neg_output, neg_labels


class Loss(tf.keras.Model):
    """
    A class representing the loss function for a classification and regression task.

    Args:
        num_hard (int): The number of hard examples to mine during training. Defaults to 0.

    Attributes:
        sigmoid (tf.keras.activations.Sigmoid): The sigmoid activation function.
        classify_loss (tf.keras.losses.BinaryCrossentropy): The binary cross entropy loss function.
        regress_loss (tf.keras.loss.Huber): The Huber loss function.
        num_hard (int): The number of hard examples to mine during training.

    Methods:
        forward(output, labels, train=True): Computes the loss given the model's output and the ground truth labels.

    Example:
        loss = Loss(num_hard=10)
        output = model(inputs)
        labels = ground_truth_labels
        loss_values = loss.forward(output, labels, train=True)
    """
    def __init__(self, num_hard=0):
        super(Loss, self).__init__()
        self.sigmoid = tf.keras.activations.Sigmoid()
        self.classify_loss = tf.keras.losses.BinaryCrossentropy()
        self.regress_loss = tf.keras.loss.Huber()
        self.num_hard = num_hard

    def forward(self, output, labels, train=True):
        batch_size = labels.size(0)
        output = output.view(-1, 5)
        labels = labels.view(-1, 5)

        pos_idcs = labels[:, 0] > 0.5
        pos_idcs = pos_idcs.unsqueeze(1).expand(pos_idcs.size(0), 5)
        pos_output = output[pos_idcs].view(-1, 5)
        pos_labels = labels[pos_idcs].view(-1, 5)

        neg_idcs = labels[:, 0] < -0.5
        neg_output = output[:, 0][neg_idcs]
        neg_labels = labels[:, 0][neg_idcs]

        if self.num_hard > 0 and train:
            neg_output, neg_labels = hard_mining(neg_output, neg_labels, self.num_hard * batch_size)
        neg_prob = self.sigmoid(neg_output)

        if len(pos_output) > 0:
            pos_prob = self.sigmoid(pos_output[:, 0])
            pz, ph, pw, pd = pos_output[:, 1], pos_output[:, 2], pos_output[:, 3], pos_output[:, 4]
            lz, lh, lw, ld = pos_labels[:, 1], pos_labels[:, 2], pos_labels[:, 3], pos_labels[:, 4]

            regress_losses = [
                self.regress_loss(pz, lz),
                self.regress_loss(ph, lh),
                self.regress_loss(pw, lw),
                self.regress_loss(pd, ld)]
            regress_losses_data = [l.data[0] for l in regress_losses]
            classify_loss = 0.5 * self.classify_loss(
                pos_prob, pos_labels[:, 0]) + 0.5 * self.classify_loss(
                neg_prob, neg_labels + 1)
            pos_correct = (pos_prob.data >= 0.5).sum()
            pos_total = len(pos_prob)

        else:
            regress_losses = [0, 0, 0, 0]
            classify_loss = 0.5 * self.classify_loss(
                neg_prob, neg_labels + 1)
            pos_correct = 0
            pos_total = 0
            regress_losses_data = [0, 0, 0, 0]
        classify_loss_data = classify_loss.data[0]

        loss = classify_loss
        for regress_loss in regress_losses:
            loss += regress_loss

        neg_correct = (neg_prob.data < 0.5).sum()
        neg_total = len(neg_prob)

        return [loss, classify_loss_data] + regress_losses_data + [pos_correct, pos_total, neg_correct, neg_total]


class GetPBB(object):
    """
    This class provides functionality to process output from a convolutional neural network for object detection.
    """
    def __init__(self, config):
        self.stride = config['stride']
        self.anchors = np.asarray(config['anchors'])

    def __call__(self, output, thresh=-3, ismask=False):
        stride = self.stride
        anchors = self.anchors
        output = np.copy(output)
        offset = (float(stride) - 1) / 2
        output_size = output.shape
        oz = np.arange(offset, offset + stride * (output_size[0] - 1) + 1, stride)
        oh = np.arange(offset, offset + stride * (output_size[1] - 1) + 1, stride)
        ow = np.arange(offset, offset + stride * (output_size[2] - 1) + 1, stride)

        output[:, :, :, :, 1] = oz.reshape((-1, 1, 1, 1)) + output[:, :, :, :, 1] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 2] = oh.reshape((1, -1, 1, 1)) + output[:, :, :, :, 2] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 3] = ow.reshape((1, 1, -1, 1)) + output[:, :, :, :, 3] * anchors.reshape((1, 1, 1, -1))
        output[:, :, :, :, 4] = np.exp(output[:, :, :, :, 4]) * anchors.reshape((1, 1, 1, -1))
        mask = output[..., 0] > thresh
        xx, yy, zz, aa = np.where(mask)

        output = output[xx, yy, zz, aa]
        if ismask:
            return output, [xx, yy, zz, aa]
        else:
            return output


def nms(output, nms_th):
    """Performs non-maximum suppression on the given output.

    :param output: The output to apply non-maximum suppression on.
    :type output: numpy.array

    :param nms_th: The threshold for non-maximum suppression.
    :type nms_th: float

    :return: The results of non-maximum suppression.
    :rtype: numpy.array
    """
    if len(output) == 0:
        return output

    output = output[np.argsort(-output[:, 0])]
    bboxes = [output[0]]

    for i in np.arange(1, len(output)):
        bbox = output[i]
        flag = 1
        for j in range(len(bboxes)):
            if iou(bbox[1:5], bboxes[j][1:5]) >= nms_th:
                flag = -1
                break
        if flag == 1:
            bboxes.append(bbox)

    bboxes = np.asarray(bboxes, np.float32)
    return bboxes


def iou(box0, box1):
    """
    Calculate the intersection over union (IoU) between two bounding boxes.

    :param box0: The coordinates and size of the first bounding box in the format [x, y, z, size].
    :param box1: The coordinates and size of the second bounding box in the format [x, y, z, size].
    :return: The IoU value.

    """
    r0 = box0[3] / 2
    s0 = box0[:3] - r0
    e0 = box0[:3] + r0

    r1 = box1[3] / 2
    s1 = box1[:3] - r1
    e1 = box1[:3] + r1

    overlap = []
    for i in range(len(s0)):
        overlap.append(max(0, min(e0[i], e1[i]) - max(s0[i], s1[i])))

    intersection = overlap[0] * overlap[1] * overlap[2]
    union = box0[3] * box0[3] * box0[3] + box1[3] * box1[3] * box1[3] - intersection
    return intersection / union


def acc(pbb, lbb, conf_th, nms_th, detect_th):
    """
    :param pbb: An array of predicted bounding boxes. Each bounding box is represented by a list of coordinates [x1, y1, x2, y2] and a confidence score.
    :param lbb: An array of ground truth bounding boxes. Each bounding box is represented by a list of coordinates [x1, y1, x2, y2].
    :param conf_th: The confidence threshold used to filter predicted bounding boxes.
    :param nms_th: The threshold used for non-maximum suppression.
    :param detect_th: The threshold used to determine if a predicted bounding box matches a ground truth bounding box.
    :return: A tuple containing the true positives (tp), false positives (fp), false negatives (fn), and the total number of ground truth bounding boxes (len(lbb)).

    """
    pbb = pbb[pbb[:, 0] >= conf_th]
    pbb = nms(pbb, nms_th)

    tp = []
    fp = []
    fn = []
    l_flag = np.zeros((len(lbb),), np.int32)
    for p in pbb:
        flag = 0
        bestscore = 0
        for i, l in enumerate(lbb):
            score = iou(p[1:5], l)
            if score > bestscore:
                bestscore = score
                besti = i
        if bestscore > detect_th:
            flag = 1
            if l_flag[besti] == 0:
                l_flag[besti] = 1
                tp.append(np.concatenate([p, [bestscore]], 0))
            else:
                fp.append(np.concatenate([p, [bestscore]], 0))
        if flag == 0:
            fp.append(np.concatenate([p, [bestscore]], 0))
    for i, l in enumerate(lbb):
        if l_flag[i] == 0:
            score = []
            for p in pbb:
                score.append(iou(p[1:5], l))
            if len(score) != 0:
                bestscore = np.max(score)
            else:
                bestscore = 0
            if bestscore < detect_th:
                fn.append(np.concatenate([l, [bestscore]], 0))

    return tp, fp, fn, len(lbb)


def topkpbb(pbb, lbb, nms_th, detect_th, topk=30):
    """
    :param pbb: Predicted bounding boxes. A list of bounding boxes with their confidence scores and class labels.
    :param lbb: Labeled bounding boxes. A list of annotated bounding boxes with their class labels.
    :param nms_th: Non-maximum suppression threshold. A float value between 0 and 1.
    :param detect_th: Detection threshold. A float value between 0 and 1.
    :param topk: Number of top bounding boxes to return. Default is 30.
    :return: Three lists of bounding boxes - true positives (tp), false positives (fp), and false negatives (fn).

    """
    conf_th = 0
    fp = []
    tp = []
    while len(tp) + len(fp) < topk:
        conf_th = conf_th - 0.2
        tp, fp, fn, _ = acc(pbb, lbb, conf_th, nms_th, detect_th)
        if conf_th < -3:
            break
    tp = np.array(tp).reshape([len(tp), 6])
    fp = np.array(fp).reshape([len(fp), 6])
    fn = np.array(fn).reshape([len(fn), 5])
    allp = np.concatenate([tp, fp], 0)
    sorting = np.argsort(allp[:, 0])[::-1]
    n_tp = len(tp)
    topk = np.min([topk, len(allp)])
    tp_in_topk = np.array([i for i in range(n_tp) if i in sorting[:topk]])
    fp_in_topk = np.array([i for i in range(topk) if sorting[i] not in range(n_tp)])
    #     print(fp_in_topk)
    fn_i = np.array([i for i in range(n_tp) if i not in sorting[:topk]])
    newallp = allp[:topk]
    if len(fn_i) > 0:
        fn = np.concatenate([fn, tp[fn_i, :5]])
    else:
        fn = fn
    if len(tp_in_topk) > 0:
        tp = tp[tp_in_topk]
    else:
        tp = []
    if len(fp_in_topk) > 0:
        fp = newallp[fp_in_topk]
    else:
        fp = []
    return tp, fp, fn
