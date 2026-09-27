import csv
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np

def get_true_label(filepath):
    dish = filepath.split('/')[-2].lower()
    if 'mandi' in dish: return 'non_veg'
    if 'shish' in dish: return 'non_veg'
    if 'meat' in dish: return 'non_veg'
    if 'grill' in dish: return 'non_veg'
    if 'shawarma' in dish: return 'non_veg'
    if 'okra' in dish: return 'non_veg'
    if 'cheese' in dish: return 'veg'
    if 'hummus' in dish: return 'veg'
    if 'tamees' in dish: return 'veg'
    if 'zaatar' in dish: return 'veg'
    return 'unknown'

y_true = []
y_pred = []

with open('results/restaurant_testing_predictions.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        true_label = get_true_label(row['file'])
        y_true.append(true_label)
        y_pred.append(row['prediction'])

labels = ['non_veg', 'veg']
cm = confusion_matrix(y_true, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(6, 6))
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels)
ax.set_yticklabels(labels)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
ax.xaxis.set_label_position('top')

for i in range(len(labels)):
    for j in range(len(labels)):
        text = ax.text(j, i, cm[i, j], ha="center", va="center", color="black" if cm[i, j] < cm.max()/2 else "white", fontsize=14)

plt.title('Confusion Matrix - Arabic Restaurant Dishes', pad=20)
plt.savefig('results/arabic_confusion_matrix.png')
print("Confusion matrix saved to results/arabic_confusion_matrix.png")
