import paddle
import paddle.fluid as fluid
import numpy as np
import cv2

eps = 1e-8

def SegLoss(preds, labels, ignore_index=255):
    n, c, h, w     = preds.shape
    n1, h1, w1, c1 = labels.shape
    assert h == h1, "Shape Error"
    assert w == w1, "Shape Error"

    costfunc       = fluid.layers.cross_entropy
    preds          = fluid.layers.transpose(preds, (0, 2, 3, 1))
    print(preds.shape)
    # mask统计labels中不等于ignore_index的元素的个数。
    mask = (labels!=ignore_index)
    # 将mask中每个元素的类型转换为float32。
    mask = fluid.layers.cast(mask, 'float32')

    cost = costfunc(preds, labels)
    if fluid.layers.has_nan(cost):
        print("Error, there is nan in cost")
        exit()
    elif fluid.layers.has_inf(cost):
        print("Error, there is inf in cost")
        exit()
    # cost是一个2为np矩阵，他的维数等于h * w
    cost = cost * mask
    # 求均值, 除数中的eps是一个。
    avg_cost = fluid.layers.mean(cost) / (fluid.layers.mean(mask) + eps)

    return avg_cost

def main():
    label = cv2.imread('./dummy_data/GroundTruth_trainval_png/2008_000002.png')
    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.int64)
    pred = np.random.uniform(0, 1, (1, 59, label.shape[0], label.shape[1])).astype(np.float32)
    label = label[:,:,np.newaxis]
    label = label[np.newaxis, :, :, :]

    with fluid.dygraph.guard(fluid.CPUPlace()):
        pred = fluid.dygraph.to_variable(pred)
        label = fluid.dygraph.to_variable(label)
        loss = SegLoss(pred, label)
        print(loss)

if __name__ == "__main__":
    main()
