from torch import nn, load
from src.models.models import get_efficientnet_b0_model, get_efficientnet_b1_model, get_efficientnet_b2_model, get_efficientnet_b3_model, get_efficientnet_b4_model
import numpy as np
import os
from pathlib import Path



def ensemble_predict(img):

    model_b0_adam, model_bo_transforms = get_efficientnet_b0_model()
    model_b0_sgd, _ = get_efficientnet_b0_model()
    
    model_b1_adam, model_b1_transforms = get_efficientnet_b1_model()
    model_b1_sgd, _ = get_efficientnet_b1_model()
    
    model_b2_adam, model_b2_transforms = get_efficientnet_b2_model()
    model_b2_sgd, _ = get_efficientnet_b2_model()

    model_b3_adam, model_b3_transforms = get_efficientnet_b3_model()
    model_b3_sgd, _ = get_efficientnet_b3_model()

    model_b4_adam, model_b4_transforms = get_efficientnet_b4_model()
    model_b4_sgd, _ = get_efficientnet_b4_model()

    models = [model_b0_adam, model_b0_sgd,
              model_b1_adam, model_b1_sgd,
              model_b2_adam, model_b2_sgd,
              model_b3_adam, model_b3_sgd,
              model_b4_adam, model_b4_sgd]
    
    model_transforms = [model_bo_transforms, model_bo_transforms,
                        model_b1_transforms, model_b1_transforms,
                        model_b2_transforms, model_b2_transforms,
                        model_b3_transforms, model_b3_transforms,
                        model_b4_transforms, model_b4_transforms]
    
    models_dir = Path('trained_models')
    model_save_paths = [models_dir/i for i in os.listdir(models_dir)]

    model_predictions = []
    for model, model_transform, model_save_path in zip(models, model_transforms, model_save_paths):
        model.load_state_dict(load(model_save_path))
        
        model.eval()
        processed_img = model_transform(img)

        pred_logits = model(processed_img.unsqueeze(0))
        model_predictions.append(pred_logits.argmax().item())
    
    ensemble_prediction = np.bincount(model_predictions).argmax().item()

    mapping = {0: 'Good',
               1: 'Fair',
               2: 'Poor',
               3: 'Severe'}
    
    condition_state_votes = {list(mapping.values())[k]:np.bincount(model_predictions)[k].item()/10 for k in range(len(np.bincount(model_predictions)))}

    all_classes = list(mapping.values())
    all_counts = {cls: condition_state_votes.get(cls, 0) for cls in all_classes}
    
    return all_counts, mapping[ensemble_prediction]
