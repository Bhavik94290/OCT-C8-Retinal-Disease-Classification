# Model Performance Comparison

## Experimental Results

The performance of the baseline models and the proposed hybrid CNN model was evaluated using accuracy, precision, recall, and F1-score on the test dataset.

| Model | Test Accuracy | Precision | Recall | F1-score |
|---|---:|---:|---:|---:|
| VGG19 | 97.93% | 97.93% | 97.93% | 97.93% |
| ResNet50 | 97.89% | 97.91% | 97.89% | 97.89% |
| DenseNet121 | 98.00% | 98.02% | 98.00% | 98.00% |
| Proposed Hybrid CNN | 97.43% | 97.43% | 97.43% | 97.43% |

## Observations

- DenseNet121 achieved the highest test accuracy among the evaluated models.
- VGG19 and ResNet50 achieved comparable performance.
- The proposed Hybrid CNN currently performs below the three baseline models.
- Further experiments are required to investigate the proposed model's performance.

## Important Note

These results should be reported only after verifying the corresponding evaluation outputs. All models should be evaluated using the same held-out test set and consistent evaluation procedures.

Precision, recall, and F1-score should be calculated using the same averaging strategy for every model.
