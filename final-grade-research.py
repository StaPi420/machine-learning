import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

with open('dataset.csv', 'r') as file:
    data = pd.read_csv(file)

    plt.hist(data['final_score'], bins=10, color='skyblue', edgecolor='black')
    plt.title('Distribution of Final Scores')
    plt.xlabel('Final Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()