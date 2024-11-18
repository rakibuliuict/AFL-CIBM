from utils.utils import *
from loss.AFLoss import AFLoss

def train(model, data_in, optim, max_epochs, model_dir, test_interval=1, start_epoch=1, device=torch.device("cuda:0")):
    best_metric = -1
    best_metric_epoch = -1
    save_loss_train = []
    save_loss_test = []
    save_metric_train = {'dice': [], 'iou': []}
    save_metric_test = {'dice': [], 'iou': []}
    
    train_loader, test_loader = data_in
  
    if start_epoch > 1:
        checkpoint_path = os.path.join(model_dir, f"model_epoch_{start_epoch-1}.pth")
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Resumed training from epoch {start_epoch-1} with model loaded from {checkpoint_path}")
        else:
            print(f"Checkpoint for epoch {start_epoch-1} not found, starting from scratch.")

    if start_epoch > 1:
        save_loss_train = np.load(os.path.join(model_dir, 'loss_train.npy')).tolist()
        save_loss_test = np.load(os.path.join(model_dir, 'loss_test.npy')).tolist()
        save_metric_train = np.load(os.path.join(model_dir, 'metric_train.npy'), allow_pickle=True).item()
        save_metric_test = np.load(os.path.join(model_dir, 'metric_test.npy'), allow_pickle=True).item()

 
    excel_file = os.path.join(model_dir, 'training_metrics.xlsx')
    if os.path.exists(excel_file):
        existing_metrics_df = pd.read_excel(excel_file)
        best_metric = existing_metrics_df['Test Dice'].max()
        best_metric_epoch = existing_metrics_df.loc[existing_metrics_df['Test Dice'].idxmax(), 'Epoch']

    for epoch in range(start_epoch, max_epochs + 1):
        print("-" * 10)
        print(f"epoch {epoch}/{max_epochs}")
        
        model.train()
        train_epoch_loss = 0
        train_step = 0
        epoch_metric_train_dice = 0
        epoch_metric_train_iou = 0
        
        for batch_data in train_loader:
            train_step += 1

            volume = batch_data["image"].to(device)
            label = batch_data["label"].to(device)
            label = label != 0

            optim.zero_grad()

            fore_pix, back_pix = count_foreground_background(label)
            volume_weight = fore_pix / (fore_pix + back_pix)

            label_5d = binarize_output(label)
            label_4d = label_5d[0]
            label_3d = label_4d[0]
            label_smoothness = compute_smoothness(label_3d)
            label_smoothness = label_smoothness.astype(np.float64)

            outputs = model(volume)

            loss = AFLoss(include_background=True, to_onehot_y=True, gamma=label_smoothness + volume_weight,
                          weight=calculate_weights(fore_pix, back_pix).to(device)) 

            train_loss = loss(outputs, label)
            train_loss.backward()
            optim.step()

            train_epoch_loss += train_loss.item()

        
            with torch.no_grad():
                dice_value = dice_metric(outputs, label)
                epoch_metric_train_dice += dice_value

                iou_value = iou_metric(outputs, label)
                epoch_metric_train_iou += iou_value

        print('-'*20)

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
            with open(os.path.join(model_dir, f'dice_metrics_epoch_{epoch}.csv'), mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['Epoch number', 'Dice Metric', 'IoU Metric'])

                model.eval()
                with torch.no_grad():
                    test_epoch_loss = 0
                    epoch_metric_test_dice = 0
                    epoch_metric_test_iou = 0
                    test_step = 0

                    for test_data in test_loader:
                        test_step += 1

                        test_volume = test_data["image"].to(device)
                        test_label = test_data["label"].to(device)
                        test_label = test_label != 0

                        test_outputs = model(test_volume)

                        test_loss = loss(test_outputs, test_label)
                        test_epoch_loss += test_loss.item()

                        test_dice_value = dice_metric(test_outputs, test_label)
                        epoch_metric_test_dice += test_dice_value

                        test_iou_value = iou_metric(test_outputs, test_label)
                        epoch_metric_test_iou += test_iou_value

                        writer.writerow([epoch, test_dice_value, test_iou_value])

                    test_epoch_loss /= test_step
                    save_loss_test.append(test_epoch_loss)
                    np.save(os.path.join(model_dir, 'loss_test.npy'), save_loss_test)

                    epoch_metric_test_dice /= test_step
                    epoch_metric_test_iou /= test_step

                    print(f'Test_Dice_metric: {epoch_metric_test_dice:.4f}')
                    print(f'Test_IoU_metric: {epoch_metric_test_iou:.4f}')

                    save_metric_test['dice'].append(epoch_metric_test_dice)
                    save_metric_test['iou'].append(epoch_metric_test_iou)

                    np.save(os.path.join(model_dir, 'metric_test.npy'), save_metric_test)

    
            model_save_path = os.path.join(model_dir, f"model_epoch_{epoch}.pth")
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

     
            if epoch_metric_test_dice > best_metric:
                best_metric = epoch_metric_test_dice
                best_metric_epoch = epoch

            print(f"Current epoch: {epoch} and current mean dice: {epoch_metric_test_dice:.4f}"
                  f" \nbest mean dice: {best_metric:.4f} at epoch: {best_metric_epoch}")

        if os.path.exists(excel_file):
            existing_metrics_df = pd.read_excel(excel_file)
            metrics_df = pd.DataFrame({
                'Epoch': [epoch],
                'Train Loss': [train_epoch_loss],
                'Test Dice': [epoch_metric_test_dice],
                'Test IoU': [epoch_metric_test_iou],
            })
            metrics_df = pd.concat([existing_metrics_df, metrics_df], ignore_index=True)
        else:
            metrics_df = pd.DataFrame({
                'Epoch': [epoch],
                'Train Loss': [train_epoch_loss],
                'Test Dice': [epoch_metric_test_dice],
                'Test IoU': [epoch_metric_test_iou],
            })

        metrics_df.to_excel(excel_file, index=False)

    print(f'Training completed. Best Dice Metric: {best_metric:.4f} at epoch: {best_metric_epoch}')


