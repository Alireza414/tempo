import utils
import os
import numpy as np
import torch
from PIL import Image
from maskrcnn_dataset import display_img_and_mask
from model import get_model_instance_segmentation
import math
import torch.utils.data
from matplotlib import pyplot as plt
from argparse import ArgumentParser
from maskrcnn_dataset import AutoCTScanDataset

def collate_fn(batch):
    return tuple(zip(*batch))

def parse_arguments(parser: ArgumentParser):
    parser.add_argument("--saved_model_path", type=str, required=False, default="saved_models/MaskRCNN_epoch_2.pt", help="Path to saved model to initialize training from")
    args = parser.parse_args()
    args_dict = vars(args)
    return args_dict

def main():
    parser: ArgumentParser = ArgumentParser()
    args_dict = parse_arguments(parser)

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = AutoCTScanDataset()
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2,collate_fn=collate_fn)

    # get the maskrcnn_model using our helper function
    maskrcnn_model = get_model_instance_segmentation(num_classes)
    # move maskrcnn_model to the right device
    maskrcnn_model.to(device)
    path_to_state_dict_file = args_dict["saved_model_path"]
    if(os.path.exists(path_to_state_dict_file)):
        maskrcnn_model.load_state_dict(torch.load(path_to_state_dict_file,map_location=torch.device(device)))
    _ = maskrcnn_model.train()

    #Freeze backbone parameters
    for p in maskrcnn_model.backbone.parameters():
        p.requires_grad = False
    params = [p for p in maskrcnn_model.parameters() if p.requires_grad]

    # construct an optimizer and a learning rate scheduler
    optimizer = torch.optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.001, amsgrad=False)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95, last_epoch=-1)

    # let's train it for 10 epochs
    num_epochs = 3
    print_loss_freq = 10
    losses = []
    mask_losses = []

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        _epoch_loss = 0
        it=0
        for i,data in enumerate(data_loader):
            imgs,trgs = data
            images = [torch.tensor(image, dtype = torch.float32).to(device) for image in imgs]
            targets = [{k: torch.tensor(v).to(device) for k, v in trg.items()} for trg in trgs]

            loss_dir = maskrcnn_model(images,targets)
            mask_loss = loss_dir['loss_mask'].item()
            summed_loss = sum(loss for loss in loss_dir.values())
            summed_loss_value = summed_loss.item()
            if(i%print_loss_freq == 0):
                print("Epoch "+str(epoch)+"/"+str(num_epochs)+", iteration "+str(i)+": Summed Loss "+str(summed_loss_value)
                      +", Mask loss "+str(mask_loss))

            if math.isfinite(summed_loss_value):
                losses.append(summed_loss_value)
                mask_losses.append(mask_loss)
                optimizer.zero_grad()
                summed_loss.backward()
                #torch.nn.utils.clip_grad_norm_(params, 10)
                optimizer.step()
                _epoch_loss+=summed_loss_value
            else:
                print('Loss is undefined:'+str(summed_loss_value)+'   skipping BackProp for step no:'+str(i))

        #Saving the maskrcnn_model after every epoch
        torch.save(maskrcnn_model.state_dict(), "saved_models/MaskRCNN_epoch_'+str(epoch)+'_latest.pt")
        print('Model saved after '+str(epoch)+" epochs")
        torch.cuda.empty_cache()

    #Plot the loss
    plt.xlabel("Iterations across epochs")
    plt.ylabel("Loss")
    plt.plot(losses, label='Summed loss')  # Plot the chart
    plt.plot(mask_losses, label='Mask Loss')  # Plot the chart
    plt.legend()
    plt.savefig('saved_model/training_loss_graph.png')


if __name__ == "__main__":
    main()
