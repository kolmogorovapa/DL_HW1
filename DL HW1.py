# Задача: даны результаты студенческой оценки преподавания, где были оценены от 1 до 10: понятность материала,
# новизна материала, полезность для общего кругозора, считаем, насколько курс подойдет студентам
# магистерской программы "языковые технологии в бизнесе и образовании"

import torch

course_1 = torch.tensor([[0.1, 0.3, 0.4]])
course_2 = torch.tensor([[0.3, 0.5, 0.1]])
course_3 = torch.tensor([[0.7, 0.9, 0.2]])
course_4 = torch.tensor([[0.5, 0.5, 0.5]])
course_5 = torch.tensor([[0.2, 0.1, 0.3]])
course_6 = torch.tensor([[0.4, 0.6, 0.8]])

dataset = [
    (course_1, torch.tensor([[0.5]])),
    (course_2, torch.tensor([[0.2]])),
    (course_3, torch.tensor([[0.8]])),
    (course_4, torch.tensor([[0.4]])),
    (course_5, torch.tensor([[0.3]])),
    (course_6, torch.tensor([[0.9]]))
]

torch.manual_seed(1200)

weights = torch.rand((1, 3), requires_grad=True)   # поскольку у нас теперь 3 параметра
bias = torch.rand((1, 1), requires_grad=True)

mse_loss_fn = torch.nn.MSELoss()

optimizer = torch.optim.SGD([weights, bias], lr=1e-3)

def predict_possibility(obj: torch.Tensor) -> torch.Tensor:
    return obj @ weights.T + bias

def calc_loss(predicted_value: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
    return mse_loss_fn(predicted_value, ground_truth)

num_epochs = 15
for i in range(num_epochs):
    for x, y in dataset:
        optimizer.zero_grad()
        possibility = predict_possibility(x)
        loss = calc_loss(possibility, y)
        loss.backward()
        print(loss)
        optimizer.step()
