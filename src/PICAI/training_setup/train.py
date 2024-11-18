from training_setup.utils.utils import *
from training_setup.Loss.AFLoss import *


import torch
import numpy as np
import os
import csv
import pandas as pd

def train(model, data_in, optim, max_epochs, model_dir, test_interval=1, device=torch.device("cuda:0")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = {'dice': [], 'iou': []}
    save_metric_test = {'dice': [], 'iou': []}
    
    train_loader, test_loader = data_in

    for epoch in range(max_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{max_epochs}")
        
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train_dice = 0
        epoch_metric_train_iou = 0
        
        for batch_data in train_loader:
            train_step += 1

            volume = batch_data["img"]
            label = batch_data["seg"]
            label = label != 0
            volume, label = (volume.to(device), label.to(device))

            optim.zero_grad()

            
            fore_pix, back_pix = count_foreground_background(label)
            volume_weight = (fore_pix) / ((fore_pix) + (back_pix))
            label_5d = binarize_output(label)
            label_4d = label_5d[0]

            label_smoothness = compute_smoothness(label_4d[0]).astype(np.float64)
            
            outputs = model(volume)

            loss = AFLoss(include_background=True, to_onehot_y=True,
                          gamma=label_smoothness + volume_weight,
                          weight=calculate_weights(fore_pix, back_pix).to(device))

            
            
            train_loss = loss(outputs, label)
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()

            # Calculate metrics
            with torch.no_grad():
                dice_value = dice_metric(outputs, label)
                epoch_metric_train_dice += dice_value

                iou_value = iou_metric(outputs, label)
                epoch_metric_train_iou += iou_value

        print('-' * 20)

     
        train_epoch_loss /= train_step
        print(f'Epoch_loss: {train_epoch_loss:.4f}')
        save_loss_train.append(train_epoch_loss)
        np.save(os.path.join(model_dir, 'loss_train.npy'), save_loss_train)

        epoch_metric_train_dice /= train_step
        epoch_metric_train_iou /= train_step

        save_metric_train['dice'].append(epoch_metric_train_dice)
        save_metric_train['iou'].append(epoch_metric_train_iou)

        np.save(os.path.join(model_dir, 'metric_train.npy'), save_metric_train)

     
        if (epoch + 1) % test_interval == 0:
            with open(os.path.join(model_dir, f'dice_metrics_epoch_{epoch+1}.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch number', 'Patient ID', 'Dice Metric', 'IoU Metric'])

                model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    epoch_metric_test_dice = 0
                    epoch_metric_test_iou = 0
                    test_step = 0

                    for test_data in test_loader:
                        test_step += 1

                        test_volume = test_data["img"]
                        test_label = test_data["seg"]
                        test_label = test_label != 0
                        test_volume, test_label = (test_volume.to(device), test_label.to(device))

                        test_outputs = model(test_volume)

                        test_loss = loss(test_outputs, test_label)
                        test_epoch_loss += test_loss.item()

                        test_dice_value = dice_metric(test_outputs, test_label)
                        epoch_metric_test_dice += test_dice_value

                        test_iou_value = iou_metric(test_outputs, test_label)
                        epoch_metric_test_iou += test_iou_value

                        patient_id = test_data["patient_id"]
                        writer.writerow([epoch + 1, patient_id, test_dice_value, test_iou_value])

                    test_epoch_loss /= test_step
                    print(f'Test_loss_epoch: {test_epoch_loss:.4f}')
                    save_loss_test.append(test_epoch_loss)
                    np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                    epoch_metric_test_dice /= test_step
                    epoch_metric_test_iou /= test_step

                    print(f'Test_Dice_metric: {epoch_metric_test_dice:.4f}')
                    print(f'Test_IoU_metric: {epoch_metric_test_iou:.4f}')

                    save_metric_test['dice'].append(epoch_metric_test_dice)
                    save_metric_test['iou'].append(epoch_metric_test_iou)

                    np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

       
            if epoch_metric_test_dice > best_metric:
                best_metric = epoch_metric_test_dice
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), os.path.join(model_dir, "best_metric_model.pth"))
                print("saved new best metric model")

            print(
                f"Current epoch: {epoch + 1} and current mean dice: {epoch_metric_test_dice:.4f}"
                f" \nbest mean dice: {best_metric:.4f} "
                f"at epoch: {best_metric_epoch}"
            )

    metrics_df = pd.DataFrame({
        'Epoch': range(1, max_epochs + 1),
        'Train Loss': save_loss_train,
        'Test Dice': save_metric_test['dice'],
        'Test IoU': save_metric_test['iou']
    })
    metrics_df.to_excel(os.path.join(model_dir, 'training_metrics.xlsx'), index=False)

    print(f'Train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}')

