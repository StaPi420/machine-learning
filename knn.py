import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from PSL_knn import find_final_grade

data = None

with open('dataset.csv', 'r') as file:
    data = pd.read_csv(file)

face_grade = int(input("Введите оценку лица (0-10): "))
body_grade = int(input("Введите оценку тела (0-10): "))
height = float(input("Введите рост (в см): ")) * (1 / 35)
penis_length = float(input("Введите длину пениса (в см): ")) * 0.5
charisma = int(input("Введите харизму (0-10): "))
has_apartment = int(input("Введите наличие квартиры (0-1): ")) * 10
has_car = int(input("Введите наличие автомобиля (0-1): ")) * 10
played_hearts_of_iron_4 = int(input("Введите играли ли вы в hoi4 (0-1): ")) * 10

final_grade = find_final_grade(data, np.array([face_grade, body_grade, height, penis_length, charisma, has_apartment, has_car, played_hearts_of_iron_4]), k = 10)
print(f"Предсказанная оценка: {final_grade}")