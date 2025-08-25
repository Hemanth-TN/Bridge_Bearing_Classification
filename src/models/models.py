from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights,\
      efficientnet_b1, EfficientNet_B1_Weights, \
        efficientnet_b2, EfficientNet_B2_Weights, \
            efficientnet_b3, EfficientNet_B3_Weights, \
                efficientnet_b4, EfficientNet_B4_Weights
from torch import nn
import torch
from torcheval.metrics.functional import multiclass_f1_score
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics.functional import multiclass_confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from PIL import Image
import io
import numpy as np
from pathlib import Path
import os



def get_efficientnet_b0_model():
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    #freezing the feature head of models
    for p in model.features.parameters():
        p.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                 nn.Linear(in_features=1280, out_features=4, bias=True))
    for p in model.features[7:].parameters():
        p.requires_grad = True
    
    for p in model.classifier.parameters():
        p.requires_grad = True

    model_transforms = EfficientNet_B0_Weights.DEFAULT.transforms()

    return model, model_transforms

def get_efficientnet_b1_model():
    model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)

    #freezing the feature head of models
    for p in model.features.parameters():
        p.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                 nn.Linear(in_features=1280, out_features=4, bias=True))
    for p in model.features[7:].parameters():
        p.requires_grad = True
    
    for p in model.classifier.parameters():
        p.requires_grad = True
    model_transforms = EfficientNet_B1_Weights.DEFAULT.transforms()

    return model, model_transforms

def get_efficientnet_b2_model():
    model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)

    #freezing the feature head of models
    for p in model.features.parameters():
        p.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                 nn.Linear(in_features=1408, out_features=4, bias=True))
    
    for p in model.features[7:].parameters():
        p.requires_grad = True
    
    for p in model.classifier.parameters():
        p.requires_grad = True
    model_transforms = EfficientNet_B2_Weights.DEFAULT.transforms()

    return model, model_transforms

def get_efficientnet_b3_model():
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    #freezing the feature head of models
    for p in model.features.parameters():
        p.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                 nn.Linear(in_features=1536, out_features=4, bias=True))

    for p in model.features[7:].parameters():
        p.requires_grad = True
    
    for p in model.classifier.parameters():
        p.requires_grad = True
    model_transforms = EfficientNet_B3_Weights.DEFAULT.transforms()

    return model, model_transforms

def get_efficientnet_b4_model():
    model = efficientnet_b4(weights=EfficientNet_B4_Weights.DEFAULT)

    for p in model.features.parameters():
        p.requires_grad = False
    
    model.classifier = nn.Sequential(nn.Dropout(p=0.5, inplace=True),
                                     nn.Linear(in_features=1792, out_features=4, bias=True))

    for p in model.features[7:].parameters():
        p.requires_grad = True
    
    for p in model.classifier.parameters():
        p.requires_grad = True
        
    model_transforms = EfficientNet_B4_Weights.DEFAULT.transforms()
    return model, model_transforms

def train_model(model, loss_fn, optimizer, train_dataloader, valid_dataloader, device, experiment_name: str, epochs = 25):
    model = model.to(device)

    train_loss_list , train_acc_list = [], []
    valid_loss_list, valid_accuracy_list = [], []

    writer = SummaryWriter(log_dir=f"runs_augmented_2/{experiment_name}")

    
    for epoch in range(epochs):
        train_data_preds, train_data_labels = [], []
        valid_data_preds, valid_data_labels = [], []

        model.train()
        train_loss, train_accuracy = 0, 0
        for X, y in train_dataloader:

            X, y = X.to(device), y.to(device)
            pred_logits = model(X)
            y_pred = torch.argmax(pred_logits, dim=1)
            
            train_data_preds.extend(y_pred.cpu().tolist())
            train_data_labels.extend(y.cpu().tolist())
            # print(pred_logits.dtype, y.dtype)

            loss = loss_fn(pred_logits, y)
            train_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_accuracy += (y_pred == y).sum().item()
        
        train_loss /= len(train_dataloader)
        train_accuracy /= len(train_dataloader.dataset)

        train_loss_list.append(train_loss)
        train_acc_list.append(train_accuracy)


        train_data_f1_score = multiclass_f1_score(torch.tensor(train_data_preds), 
                                                  torch.tensor(train_data_labels), 
                                                  num_classes=4, average='macro').item()


        valid_loss, valid_accuracy = 0,0


        model.eval()
        for X, y in valid_dataloader:
            X, y = X.to(device), y.to(device)
            with torch.no_grad():
                pred_logits = model(X)
                y_pred = torch.argmax(pred_logits, dim=1)
                valid_data_preds.extend(y_pred.cpu().numpy())
                valid_data_labels.extend(y.cpu().numpy())

                loss = loss_fn(pred_logits, y)
                valid_loss += loss.item()

                valid_accuracy += (y_pred == y).sum().item()
            
        valid_loss /= len(valid_dataloader)
        valid_accuracy /= len(valid_dataloader.dataset)

        valid_loss_list.append(valid_loss)
        valid_accuracy_list.append(valid_accuracy)

        valid_data_f1_score = multiclass_f1_score(torch.tensor(valid_data_preds), 
                                                  torch.tensor(valid_data_labels), 
                                                  num_classes=4, average='macro').item()
        
        valid_data_confusion_matrix = multiclass_confusion_matrix(torch.tensor(valid_data_preds), 
                                                                  torch.tensor(valid_data_labels), num_classes=4)
        
        print(f"Epoch: {epoch} | Train accuracy: {train_accuracy:0.4f} | Train Loss: {train_loss:0.4f} | Train F1 Score Weighted: {train_data_f1_score:0.3f} | Valid accuracy: {valid_accuracy:0.4f} | Valid Loss: {valid_loss:0.4f} | Valid F1 Score Weighted: {valid_data_f1_score:0.3f}")

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/valid', valid_loss, epoch)

        writer.add_scalar('Accuracy/train', train_accuracy, epoch)
        writer.add_scalar('Accuracy/valid', valid_accuracy, epoch)

        writer.add_scalar('F1_score/train', train_data_f1_score, epoch)
        writer.add_scalar('F1_score/valid', valid_data_f1_score, epoch)

    fig, ax = plot_confusion_matrix(conf_mat=valid_data_confusion_matrix.cpu().numpy(), show_normed=True, colorbar=True, figsize=(6,6))
    plt.title(f"Last epoch Validation Accuracy: {valid_accuracy:.2%}")  # title with percentage
    plt.tight_layout()
    

    
    # Convert matplotlib figure to PIL image
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    img = Image.open(buf).convert("RGB")

    # Convert PIL image to tensor
    img_tensor = torch.tensor(np.array(img)).permute(2, 0, 1)

    writer.add_image("Confusion Matrix", img_tensor, global_step=epochs-1)
    
    writer.close()


    return {"train_loss_list": train_loss_list,
            "train_accuracy_list": train_acc_list,
            "valid_loss_list": valid_loss_list,
            "valid_accuracy_list": valid_accuracy_list}



def test_models(models, test_dataloaders):
    # Step 1: Get all true labels from one of the dataloaders
    # It's assumed all dataloaders have the same data and labels.
    test_labels_list = []
    for _, y in test_dataloaders[0]:
        test_labels_list.extend(y.cpu().numpy())
    test_labels = np.array(test_labels_list)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Step 2: Get predictions from all models
    all_predictions = []
    for model, test_loader in zip(models, test_dataloaders):
        model_predictions = []
        model.eval()
        for images, _ in test_loader:
            # Move images to the correct device (e.g.,s 'cuda')
            images = images.to(device)
            with torch.no_grad():
                outputs = model(images)
                predicted = torch.argmax(outputs, dim=1)
                model_predictions.extend(predicted.cpu().numpy())
        all_predictions.append(model_predictions)

    # Convert to numpy array
    all_predictions = np.array(all_predictions)

    # Step 3: Implement the majority vote with the new condition
    final_predictions = np.empty_like(test_labels, dtype=int)

    for i in range(all_predictions.shape[1]):
        # Get all predictions for the i-th sample
        sample_preds = all_predictions[:, i]
        # Count the occurrences of each unique prediction
        counts = np.bincount(sample_preds)
        # Find the predicted class (the one with the highest count)
        majority_vote = np.argmax(counts)
        # Find the number of votes for that class
        vote_count = counts[majority_vote]

        if vote_count >= 3:
            final_predictions[i] = majority_vote
        else:
            final_predictions[i] = -5

    # Step 4: Calculate accuracy, ignoring the -5 predictions
    # Only count samples where a majority was found
    mask = final_predictions != -5
    if np.sum(mask) > 0:
        correct_predictions = (final_predictions[mask] == test_labels[mask]).sum()
        total_samples = np.sum(mask)
        ensemble_accuracy = correct_predictions / total_samples
    else:
        ensemble_accuracy = 0.0

    print(f"Number of samples where no majority was found (output -5): {len(final_predictions) - np.sum(mask)}")
    print(f"Ensemble accuracy with majority vote and confidence threshold: {ensemble_accuracy:.4f}")

    return final_predictions, all_predictions


