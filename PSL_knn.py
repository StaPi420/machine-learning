import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dataset_generator import generate_dataset


def find_final_grade(learning_data: pd.DataFrame, instance: np.array, k=10):
    distances = []

    for item in learning_data.to_dict('records'):
        distance = np.linalg.norm(instance - np.array([item[key] for key in item if key != 'final_score']))
        distances.append((distance, item['final_score']))

    distances.sort(key=lambda x: x[0])
    k_nearest = distances[:k]

    numerator = sum([score / distance for distance, score in k_nearest])
    denominator = sum([1 / distance for distance, score in k_nearest])

    return numerator / denominator if denominator != 0 else np.mean([score for _, score in k_nearest])

def testing_model():
    learning_data, testing_data = generate_dataset(1000)
    actual_grades = []
    predicted_grades = []

    for test in testing_data.to_dict('records'):
        #int(f"Testing instance: {test}")
        predicted_grade = find_final_grade(learning_data, np.array([test[key] for key in test if key != 'final_score']), k=5)
        #print(f"Predicted: {predicted_grade}, Actual: {test['final_score']}")
        predicted_grades.append(predicted_grade)
        actual_grades.append(test['final_score'])

    actual_grades = np.array(actual_grades)
    predicted_grades = np.array(predicted_grades)
    errors = predicted_grades - actual_grades
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    total_variance = np.sum((actual_grades - np.mean(actual_grades)) ** 2)
    r_squared = (
        1 - np.sum(errors ** 2) / total_variance
        if total_variance != 0
        else np.nan
    )

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R-squared: {r_squared:.3f}")

    plt.scatter(actual_grades, predicted_grades, color='royalblue', label='Test samples')
    if actual_grades.size > 0:
        grade_min = min(min(actual_grades), min(predicted_grades))
        grade_max = max(max(actual_grades), max(predicted_grades))
        plt.plot([grade_min, grade_max], [grade_min, grade_max], 'r--', label='Perfect prediction')

    plt.xlabel('Actual grade')
    plt.ylabel('Predicted grade')
    plt.title('KNN: predicted vs actual grades')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

