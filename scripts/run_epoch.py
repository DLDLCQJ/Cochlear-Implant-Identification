import torch
import torch.nn as nn

def run_epoch(args, model,  dataloader, criterion,optimizer, lr_scheduler, fold, epoch, is_training=True,is_testing=False):
    model.train() if is_training else model.eval()
    epoch_loss, total,epoch_correct = 0, 0, 0
    predictions, truths =[], []
    for i, (images, labels) in enumerate(dataloader):
        # Reshape images to (batch_size, input_size) and then move to device
        images,labels = images.to(args.device),labels.to(args.device)
        outputs = model(images).squeeze(dim=1)
        predicts = (torch.sigmoid(outputs) > 0.5).float()
        labels = labels.to(torch.float32)
        loss= criterion(outputs, labels)
        loss = loss / args.gradient_accumulation_steps
        ##
        if is_training:
            loss.backward()
            if (i + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()        # Update network parameters
                lr_scheduler.step()     # Update learning rate
                optimizer.zero_grad()   # Reset gradient
        # Print statistics
        epoch_loss += loss.item()*images.size(0)
        total += labels.size(0)
        epoch_correct += (predicts == labels).sum().item()
        predictions.extend(outputs.detach().cpu().numpy())
        truths.extend(labels.cpu().numpy())
        if is_training:
            print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.num_epochs}], Val_Loss: {epoch_loss:.4f}, Val_Acc: {epoch_correct / total:.4f}")   
        elif is_testing:
            print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.num_epochs}], Te_Loss: {epoch_loss:.4f}, Te_Acc: {epoch_correct / total:.4f}")   
        else:
            print(f"Fold {fold + 1}, Epoch [{epoch + 1}/{args.num_epochs}], Tr_Loss: {epoch_loss:.4f}, Tr_Acc: {epoch_correct / total:.4f}")
        print(50*"--")
    return epoch_loss, predictions, truths
