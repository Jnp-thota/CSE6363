from assignments.assignment1.BatchProcessing import BatchProcessing
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np



iris = load_iris()
X =iris.data
y = iris.target


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.1,stratify=y,random_state=42)

unique_classes, class_counts = np.unique(y_test, return_counts=True)
main_class_counts = min(class_counts)

selected_indices = []

for class_label in unique_classes:
    class_indices = np.where(y_test ==  class_label)[0]
    selected_indices.extend(class_indices[:main_class_counts])

X_test = X_test[selected_indices]
y_test = y_test[selected_indices]

model = BatchProcessing(batch_size=32,max_epochs=100, patience=3)
model.fit(X_train, y_train)

# Plot the loss history
model.plot_loss_history()