import torch

x_data = torch.tensor([-1, 0, 2])
y_data = torch.tensor([2, 0, 0])

num_datapoints = len(x_data)

sum_product_xy = torch.sum(torch.tensor([x_data[i] * y_data[i] for i in range(0, num_datapoints)]))
sum_x = torch.sum(x_data)
sum_x_squared = torch.sum(x_data ** 2) # sum(x^2)
sum_y = torch.sum(y_data)

print(sum_product_xy)
print(sum_x_squared)

# Calculating slope and intercept using a formula (When using the sum of squared errors [SSE])
beta_1 = ((num_datapoints * sum_product_xy) - (sum_x * sum_y)) / ((num_datapoints * (sum_x_squared)) - (sum_x ** 2))
beta_0 = ((sum_x_squared * sum_y) - (sum_x * sum_product_xy)) / ((num_datapoints * (sum_x_squared)) - (sum_x ** 2))

# y_pred = beta_0 + beta_1(x)
print(beta_1)
print(beta_0)