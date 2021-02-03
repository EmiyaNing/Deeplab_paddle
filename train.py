import os
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
import numpy as np
import argparse
from utils import AverageMeter
from deeplab import DeepLab
from dataload import Dataloader,Transform
from Lossfunc import SegLoss

parser = argparse.ArgumentParser()
parser.add_argument('--net', type=str, default='deeplab')
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--image_folder', type=str, default='./dummy_data')
parser.add_argument('--image_list_file', type=str, default='./dummy_data/list.txt')
parser.add_argument('--checkpoint_folder', type=str, default='./output')
parser.add_argument('--save_freq', type=int, default=2)

args = parser.parse_args()

def train(dataloader, model, costfunction, optimizer, epoch, total_batch):
    model.train()
    train_loss_meter = AverageMeter()
    for batch_id, data in enumerate(dataloader):
        image = data[0].astype("float32")
        label = data[1]
        image = fluid.layers.transpose(image, perm=(0, 3, 1, 2))

        pred  = model(image)
        loss  = costfunction(pred, label)
        loss.backward()
        optimizer.minimize(loss)
        model.clear_gradients()

        n = image.shape[0]
        train_loss_meter.update(loss.numpy()[0], n)
        print(f"Epoch[{epoch:03d}/{args.num_epochs:03d}], " +
                f"Step[{batch_id:04d}/{total_batch:04d}], " +
                f"Average Loss: {train_loss_meter.avg:4f}")

    return train_loss_meter.avg

def main():
    Place = paddle.fluid.CPUPlace()
    with fluid.dygraph.guard(Place):
        transform  = Transform(256)
        dataload   = Dataloader(args.image_folder, args.image_list_file, transform, True)
        train_load = fluid.io.DataLoader.from_generator(capacity=1 , use_multiprocess=False)
        train_load.set_sample_generator(dataload, batch_size=args.batch_size, places=Place)
        total_batch= int(len(dataload) / args.batch_size)

        if args.net == 'deeplab':
            model = DeepLab(59)
        else:
            print("Other model haven't finished....")

        costFunc = SegLoss
        adam     = AdamOptimizer(learning_rate=args.lr, parameter_list=model.parameters())

        for epoch in range(1, args.num_epochs + 1):
            train_loss = train(train_load, model, costFunc, adam, epoch, total_batch)
            print(f"----- Epoch[{epoch}/{args.num_epochs}] Train Loss: {train_loss}")

            if epoch % args.save_freq == 0 or epoch == args.num_epochs:
                model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{train_loss}")

                model_dict = model.state_dict()
                fluid.save_dygraph(model_dict, model_path)
                optimizer_dict = optimizer.state_dict()
                fluid.save_dygraph(optimizer_dict, model_path)
                print(f'----- Save model: {model_path}.pdparams')
                print(f'----- Save optimizer: {model_path}.pdopt')



if __name__ == "__main__":
    main()