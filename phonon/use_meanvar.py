import numpy as np
import matplotlib.pyplot as plt


def load_numbers_from_file(file_path):

    numbers = []

    with open(file_path, 'r') as file:
        for line in file:
            parts = line.split()
            if len(parts) == 2:  
                number = float(parts[1])
                numbers.append(number)
    
    return np.array(numbers)

# Example usage
file_path = '/home/yuming/Downloads/MT_1/meanvar.txt'
numbers_array = load_numbers_from_file(file_path)
numbers_array = np.sqrt(1/numbers_array)
plt.scatter(np.linspace(-1,1,num=len(numbers_array)),numbers_array)
plt.show()
