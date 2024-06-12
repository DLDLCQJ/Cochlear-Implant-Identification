import argparse
import os
import glob 
from os.path import join as opj
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.models as tvmodels
import glob 


def inference(args):
    # Define the class labels
    class_labels = ['IC_improved', 'non_IC_improved']
    confusion_matrices=[]
    import glob
    for n in range(args.k):
        # Load the pretrained model
        if args.network == 'Inception':
            model = timm.create_model('inception_v3', pretrained=True)
            in_features = model.get_classifier().in_features
            model.fc =linear_block(in_features, args.num_classes,args.hidden_size,False,nn.SELU)
        elif args.network in ['Alexnet','Mobilenet']:
            model = tvmodels.mobilenet_v2(pretrained=True, progress = False)
            model_features = model.features
            input_size = [64,3,128,128]
            avgpool = nn.AdaptiveAvgPool2d((1,1))
            in_features = avgpool(model_features(torch.rand(input_size))).shape[1]
            model.classifier = linear_block(in_features, args.num_classes,args.hidden_size,False,nn.SELU)
        ##
        model_path = glob.glob(opj(args.save_dir, f'model_fold{n+1}_epoch*_{args.network}_{args.img_file}.pth'))
        if not model_path:
            raise FileNotFoundError(f"No model found for fold {n+1}")
        print(type(model_path),model_path)
        model.load_state_dict(torch.load(model_path[0]),strict=False)
        model.eval()
        # Load and preprocess the image
        X, y = load_test_data(args)
        test_dataset = MyDataset(args.norm_type,X, y)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        # Perform inference
        with torch.no_grad():
            total, correct = 0, 0
            # Get the predicted class label and confidence scores
            truths , predictions = [], []
            for images, labels in test_loader:
                if args.network in ['Alexnet','Mobilenet']:
                    features = model_features(images)
                    outputs = model.classifier(avgpool(features).view(1,-1)).squeeze(dim=1)
                elif args.network == 'Inception':
                    outputs = model(images).squeeze(dim=1)
                predicted = (torch.sigmoid(outputs) > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                predictions.extend(outputs.cpu().numpy())
                truths.extend(labels.cpu().numpy())
        te_acc = correct / total
        print(f"Fold {n+1} - Total Sample: {total}, Total Correct: {correct}, Test Accuracy: {te_acc}")
        predictions_ = (1 / (1 + np.exp(-np.array(predictions))) > 0.5).astype(float)
        truths = np.array(truths)
        predictions_ = np.array(predictions_)
        cm = confusion_matrix(truths, predictions_,labels=[0, 1])
        confusion_matrices.append(cm)
        print(truths.shape, predictions_.shape)
        scores_dict = compute_scores(truths, predictions_)
        print(f"Fold {n+1} - Sensitivity = {scores_dict['sensitivity']}")
        print(f"Fold {n+1} - Specificity = {scores_dict['specificity']}")
        print(f"Fold {n+1} - ROC_AUC = {scores_dict['roc_auc']}")
        print(f"Fold {n+1} - Accuracy = {scores_dict['accuracy']}")
    compute_mean_metrics(confusion_matrices, truths, predictions)
if __name__ == "__main__":
    args = parse_arguments()  # You could provide default args here
    inference(args)
