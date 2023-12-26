import torch
from single_regression import SingleRegression

data1 = torch.tensor(
                    [
                    [-1, 0, 2], 
                    [2, 0, 0]
                    ]
                    )

data2 = torch.tensor(
                    [
                    [6, 5, 3, 4, 7, 10.5, 16, 200, 1025.52, 12000, 9, 11, 50, 100, 2000], 
                    [3, 2.5, 1.5, 2, 3.5, 5, 8, 99.5, 512.7, 6012, 4.5, 5.5, 25, 50.1, 999.75],
                    ]
                    )

model_1 = SingleRegression()
model_1.fit(data1)

model_2 = SingleRegression()
model_2.fit(data2)

x_preds = [1020, 2040, 5000, 9999, 3000, 10000]
_ = model_1.get_predictions(x_preds = x_preds)
_ = model_2.get_predictions(x_preds = x_preds)