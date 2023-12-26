import torch

class SingleRegression:

    def __init__(self, data):
        self.__create_model(data)

    def __create_model(self, data):

        x_data, y_data = data
        num_datapoints = len(x_data)

        sum_product_xy = torch.sum(torch.tensor([x_data[i] * y_data[i] for i in range(0, num_datapoints)]))
        sum_x = torch.sum(x_data)
        sum_x_squared = torch.sum(x_data ** 2) # sum(x^2)
        sum_y = torch.sum(y_data)

        # Calculating slope and intercept using a formula (When using the sum of squared errors [SSE])
        self.beta_1 = ((num_datapoints * sum_product_xy) - (sum_x * sum_y)) / ((num_datapoints * (sum_x_squared)) - (sum_x ** 2))
        self.beta_0 = ((sum_x_squared * sum_y) - (sum_x * sum_product_xy)) / ((num_datapoints * (sum_x_squared)) - (sum_x ** 2))

        # y_pred = beta_0 + beta_1(x)
    
    def get_prediction(self, x):

        return self.beta_0 + (self.beta_1 * torch.tensor(x))
        

