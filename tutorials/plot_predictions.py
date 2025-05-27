import json
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

# Read the JSON file
with open('/grand/GeomicVar/tarak/cpgpt/CpGPT/data/tutorials/processed/fhs_setup/embeddings/true_and_predicted.json', 'r') as f:
    data = json.load(f)

# Extract true and predicted values
# set_trace()  
true_values = data['true_conditions']
predicted_values = data['predicted_conditions']
predicted_values = np.array(predicted_values).flatten()  # Ensure it's a 1D array

# Create scatter plot
plt.figure(figsize=(10, 8))
plt.scatter(true_values, predicted_values, alpha=0.5)
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], 'r--', label='Perfect prediction')

# Calculate correlation coefficient
correlation = np.corrcoef(true_values, predicted_values)[0, 1]

# Add labels and title
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title(f'True vs Predicted Values\nCorrelation: {correlation:.3f}')
plt.legend()

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Make plot aspect ratio equal
plt.axis('equal')

# Save plot
plt.savefig('true_vs_predicted_plot.png')
plt.close()

set_trace()  