import json
import numpy as np
import matplotlib.pyplot as plt
from pdb import set_trace

# Read both JSON files
with open('/grand/GeomicVar/tarak/cpgpt/CpGPT/data/tutorials/processed/fhs_setup/embeddings/true_and_predicted.json', 'r') as f:
    data_age = json.load(f)

with open('/grand/GeomicVar/tarak/cpgpt/CpGPT/data/tutorials/processed/fhs_setup/embeddings/true_and_predicted_clock_proxies.json', 'r') as f:
    data_clock_proxies = json.load(f)

with open('/grand/GeomicVar/tarak/cpgpt/CpGPT/data/tutorials/processed/fhs_setup/embeddings/metadata_true.json', 'r') as f:
    metadata = json.load(f)

# Extract and prepare data
true_age = np.array(data_age['true_conditions'])
pred_age = np.array(data_age['predicted_conditions']).flatten()
pred_altumage = np.array([pred[0] for pred in data_clock_proxies['predicted_conditions']])
pred_dunedinpace_x100 = np.array([pred[1] for pred in data_clock_proxies['predicted_conditions']])
pred_grimage2 = np.array([pred[2] for pred in data_clock_proxies['predicted_conditions']])
pred_hrsinchphenoage = np.array([pred[3] for pred in data_clock_proxies['predicted_conditions']])
pred_pchorvath2013 = np.array([pred[4] for pred in data_clock_proxies['predicted_conditions']])

# choose the appropriate clock
# pred_clockage = pred_altumage  # or pred_dunedinpace_x100, pred_grimage2, pred_hrsinchphenoage, pred_pchorvath2013
pred_clockage = pred_pchorvath2013

# Get CHIP status
chip_status = np.array(metadata['haschip'])
chip_indices = np.where(chip_status == 1)[0]
non_chip_indices = np.where(chip_status == 0)[0]


# Calculate correlations
corr_age = np.corrcoef(true_age, pred_age)[0, 1]
corr_clockage = np.corrcoef(true_age, pred_altumage)[0, 1]

# Create plot
plt.figure(figsize=(12, 10))

# # Plot both predictions
# plt.scatter(true_age, pred_age, alpha=0.6, color='blue', marker='o', 
#            label=f'Direct Age Prediction (r={corr_age:.3f})')
# plt.scatter(true_age, pred_clockage, alpha=0.6, color='red', marker='^', 
#            label=f'Clock-Based Age Prediction (r={corr_clockage:.3f})')

# Plot non-CHIP samples
plt.scatter(true_age[non_chip_indices], pred_age[non_chip_indices], alpha=0.6, color='blue', marker='o', 
           label=f'Direct Age Prediction (r={corr_age:.3f}) Non-CHIP')
plt.scatter(true_age[non_chip_indices], pred_clockage[non_chip_indices], alpha=0.6, color='red', marker='^', 
           label=f'Clock-Based Age Prediction (r={corr_clockage:.3f}) Non-CHIP')

# Plot CHIP samples in yellow
plt.scatter(true_age[chip_indices], pred_age[chip_indices], alpha=0.6, color='yellow', marker='o', 
           label='Direct Age Prediction CHIP')
plt.scatter(true_age[chip_indices], pred_clockage[chip_indices], alpha=0.6, color='yellow', marker='^', 
           label='Clock-Based Age Prediction CHIP')

# Add perfect prediction line
plt.plot([min(true_age), max(true_age)], [min(true_age), max(true_age)], 
         'k--', label='Perfect prediction')

# Customize plot
plt.xlabel('True Age (years)', fontsize=12)
plt.ylabel('Predicted Age (years)', fontsize=12)
plt.title('Comparison of Age Prediction Methods', fontsize=14, pad=20)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=10)
plt.axis('equal')

# Add text box with statistics
stats_text = (f'Statistics:\n'
              f'Direct Age: r={corr_age:.3f}\n'
              f'Clock Age: r={corr_clockage:.3f}')
plt.text(0.05, 0.95, stats_text, transform=plt.gca().transAxes, 
         bbox=dict(facecolor='white', alpha=0.8), fontsize=10, 
         verticalalignment='top')

# Save plot with higher resolution
plt.savefig('age_predictions_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Print correlations
print(f"Direct age prediction correlation: {corr_age:.3f}")
print(f"Clock-based age prediction correlation: {corr_clockage:.3f}")

set_trace()