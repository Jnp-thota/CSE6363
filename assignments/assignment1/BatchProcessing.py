import numpy as np
import pickle 
import matplotlib.pyplot as plt
class BatchProcessing:
    def __init__(self, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Linear Regression using Gradient Descent.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y, batch_size=32, regularization=0, max_epochs=100, patience=3):
        """Fit a linear model.

        Parameters:
        -----------
        batch_size: int
            The number of samples per batch.
        regularization: float
            The regularization parameter.
        max_epochs: int
            The maximum number of epochs.
        patience: int
            The number of epochs to wait before stopping if the validation loss
            does not improve.
        """
        self.batch_size = batch_size
        self.regularization = regularization
        self.max_epochs = max_epochs
        self.patience = patience

        # TODO: Initialize the weights and bias based on the shape of X and y.
        self.weights = np.random.randn(X.shape[1])
        self.bias = np.random.randn(1)

        best_weights = self.weights
        best_bias = self.bias
        best_val_loss = float('inf')
        patience_counter =0

        # TODO: Implement the training loop.
        # set aside 10% of the training data as a validation set
        val_size = int(0.1 * len(X))
        X_val, y_val = X[-val_size:], y[-val_size:]
        X_train, y_train = X[:-val_size], y[:-val_size]

        for epoch in range(max_epochs):
            indices = np.random.permutation(len(X_train))
            X_train_shuffled = X_train[indices]
            y_train_shuffled = y_train[indices]

            for i in range(0,len(X_train),batch_size):
                X_batch = X_train_shuffled[i:i+batch_size]
                y_batch = y_train_shuffled[i:i+batch_size]

                self.gradient_descent(X_batch,y_batch,regularization)

            train_loss = self.mean_squared_errors(X_train,y_train,regularization)
            val_loss = self.mean_squared_errors(X_val,y_val,regularization)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights =self.weights
                best_bias = self.bias
                patience_counter = 0
            else:
                patience_counter+=1
            
            if patience_counter>= patience:
                print(f"Early stopping at epoch {epoch+1} with validation loss {val_loss}")
                break

            self.loss_history.append(train_loss)

            print(f"Epoch {epoch+1}/{max_epochs}- Training Loss: {train_loss}, Validation Loss: {val_loss}")

            self.weights = best_weights
            self.bias = best_bias
    
    def mean_squared_errors(self,X,y,regularization):
        predictions = self.predict(X)
        mse= np.mean((predictions-y)**2)

        regularization_term = regularization*np.sum(self.weights**2)
        mse += regularization_term
        
        return mse

    def gradient_descent(self,X_batch,y_batch,regularization):
        predictions = self.predict(X_batch)
        errors = predictions-y_batch
        d_weights = 2 * np.dot(errors, X_batch) / len(X_batch) + 2 * regularization * self.weights
        d_bias = 2 * np.sum(errors) / len(X_batch)

        # Update weights and bias
        self.weights =d_weights
        self.bias =d_bias


         

    def predict(self, X):
        """Predict using the linear model.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        """
       
        # # TODO: Implement the prediction function.
        return np.dot(X,self.weights) + self.bias

    def score(self, X, y):
        """Evaluate the linear model using the mean squared error.

        Parameters
        ----------
        X: numpy.ndarray
            The input data.
        y: numpy.ndarray
            The target data.
        """
        # TODO: Implement the scoring function.
        predictions = self.predict(X)
        mse = np.mean((predictions-y)**2)
        return mse
    
    def save(self, file_path):
        with open(file_path,'wb') as file:
            params ={'weights' : self.weights, 'bias' : self.bias}
            pickle.dump(params,file)
     
    def load(self,file_path):
        with open(file_path,'rb') as file:
            params = pickle.laod(file)
            self.weights = params['weights']
            self.bias = params['bias']
    
    def plot_loss_history(self):
        plt.plot(self.loss_history)
        plt.xlabel('Training steps')
        plt.ylabel('Mean Squared Errors')
        plt.title("Training Loss")
        plt.show()



from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np

#from assignments.assignment1.LinearRegression import LinearRegression

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

