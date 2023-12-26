import torch
from single_regression import SingleRegression

data = torch.tensor(
                    [
                    [-1, 0, 2], 
                    [2, 0, 0]
                    ]
                    )

model = SingleRegression(data = data)

x_preds = [5]
y_preds = [model.get_prediction(x_pred) for x_pred in x_preds]
for i in range(0, len(y_preds)):
    print(f"x_pred: {x_preds[i]} | Prediction (y_pred): {y_preds[i]}")
