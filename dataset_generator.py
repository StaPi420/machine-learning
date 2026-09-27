import numpy as np
import pandas as pd
import csv

def generate_dataset(num_samples):
    """
    Generates a synthetic dataset with the specified number of samples.

    Parameters:
        num_samples (int): The number of samples to generate.

    Returns:
        pandas.DataFrame: A DataFrame containing the generated dataset.
    """

    data = pd.DataFrame({
        'face_grade': np.clip(np.random.normal(loc=6, scale=3, size=num_samples), 0, 10),
        'body_grade': np.clip(np.random.normal(loc=6, scale=3, size=num_samples), 0, 10),
        'height': np.random.normal(loc=175, scale=10, size=num_samples) * (1/ 35),
        'penis_length': np.random.normal(loc=15, scale=3, size=num_samples) * 0.5,
        'has_apartment': np.random.choice([0, 0, 1], size=num_samples) * 10,
        'has_car': np.random.choice([0, 1], size=num_samples) * 10,
        'played_hearts_of_iron_4': np.random.choice([0, 0, 0, 0, 1], size=num_samples) * 10,
        'charisma': np.clip(np.random.normal(loc=6, scale=3, size=num_samples), 0, 10),
    })

    # Generating and selecting learning data

    learning_data, testing_data = data[:int(0.8 * len(data))], data[int(0.8 * len(data)):]

    learning_data['final_score'] = (learning_data['face_grade'] + 
                                    learning_data['body_grade'] + 
                                    learning_data['charisma'] + 
                                    learning_data['penis_length'] +
                                    learning_data['height'] + 
                                    learning_data['has_apartment'] +
                                    learning_data['has_car'] - 
                                    learning_data['played_hearts_of_iron_4']) / 7

    testing_data['final_score'] = (testing_data['face_grade'] + 
                                   testing_data['body_grade'] + 
                                   testing_data['charisma'] + 
                                   testing_data['penis_length'] +
                                   testing_data['height'] + 
                                   testing_data['has_apartment'] +
                                   testing_data['has_car'] - 
                                   testing_data['played_hearts_of_iron_4']) / 7

    return learning_data, testing_data

def generate_csv_dataset(number_of_samples):
    with open('dataset.csv', 'w', newline='') as file:
        learning_data, testing_data = generate_dataset(number_of_samples)
        writer = csv.writer(file)
        writer.writerow(['face_grade', 'body_grade', 'height', 'penis_length', 'has_apartment', 'has_car', 'played_hearts_of_iron_4', 'charisma', 'final_score'])
        for _, row in learning_data.iterrows():
            writer.writerow(row)


generate_csv_dataset(1000)