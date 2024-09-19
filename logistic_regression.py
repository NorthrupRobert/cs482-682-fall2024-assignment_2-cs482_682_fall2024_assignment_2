import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
import argparse
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MyLogisticRegression:
    def __init__(self, dataset_num, perform_test):
        self.training_set = None
        self.test_set = None
        self.model_logistic = LogisticRegression()
        self.model_linear = LinearRegression()
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        
        self.perform_test = perform_test
        self.dataset_num = dataset_num
        self.read_csv(self.dataset_num)

    def read_csv(self, dataset_num):
        if dataset_num == '1':
            train_dataset_file = 'train_q1_1.csv'
            test_dataset_file = 'test_q1_1.csv'
        elif dataset_num == '2':
            train_dataset_file = 'train_q1_2.csv'
            test_dataset_file = 'test_q1_2.csv'
        else:
            print("unsupported dataset number")
            
        self.training_set = pd.read_csv(train_dataset_file, sep=',', header=0)
        self.X_train = self.training_set[['exam_score_1', 'exam_score_2']]
        self.y_train = self.training_set['label']
        if self.perform_test:
            self.test_set = pd.read_csv(test_dataset_file, sep=',', header=0)
            self.X_test = self.test_set[['exam_score_1', 'exam_score_2']]
            self.y_test = self.test_set['label']
        # else:
        #     print("ERROR3")
        
        
    def model_fit_linear(self):
        '''
        initialize self.model_linear here and call the fit function
        '''
        # Find the model that best fits the training data, using linear regression
        # print(f"XTRAIN SIZE: {self.X_train.shape}")
        # print(f"YTRAIN SIZE: {self.y_train.shape}")
        self.model_linear.fit(self.X_train, self.y_train)
    
    def model_fit_logistic(self):
        '''
        initialize self.model_logistic here and call the fit function
        '''
        # Find the model that best fits the training data, using linear regression
        self.model_logistic.fit(self.X_train, self.y_train)
    
    def model_predict_linear(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_linear()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_linear is not None, "Initialize the model, i.e. instantiate the variable self.model_linear in model_fit_linear method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "
        
        y_pred_continuous = None
        y_pred = None
        if self.X_test is not None:
            # perform prediction here
            y_pred_continuous = self.model_linear.predict(self.X_test)
            y_pred = np.where(y_pred_continuous > 0.5, 1, 0)

            accuracy = accuracy_score(self.y_test, y_pred)

            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)

            precision = np.array(precision)
            recall = np.array(recall)
            f1 = np.array(f1)
            support = np.array(support)
        # else:
        #     print('ERROR2')
        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

    def model_predict_logistic(self):
        '''
        Calculate and return the accuracy, precision, recall, f1, support of the model.
        '''
        self.model_fit_logistic()
        accuracy = 0.0
        precision, recall, f1, support = np.array([0,0]), np.array([0,0]), np.array([0,0]), np.array([0,0])
        assert self.model_logistic is not None, "Initialize the model, i.e. instantiate the variable self.model_logistic in model_fit_logistic method"
        assert self.training_set is not None, "self.read_csv function isn't called or the self.trianing_set hasn't been initialized "

        y_pred_continuous = None
        y_pred = None
        if self.X_test is not None:
            # perform prediction here
            y_pred_continuous = self.model_logistic.predict(self.X_test)
            y_pred = np.where(y_pred_continuous > 0.5, 1, 0)

            accuracy = accuracy_score(self.y_test, y_pred)

            precision, recall, f1, support = precision_recall_fscore_support(self.y_test, y_pred, average=None)

            precision = np.array(precision)
            recall = np.array(recall)
            f1 = np.array(f1)
            support = np.array(support)
        # else:
        #     print('ERROR1')

        
        assert precision.shape == recall.shape == f1.shape == support.shape == (2,), "precision, recall, f1, support should be an array of shape (2,)"
        return [accuracy, precision, recall, f1, support]

    def plot_decision_boundary(self, model, title):
        """
        Plots the decision boundary for a given model (logistic or linear).
        """
        # Create a mesh grid based on the range of values for the two features
        x_min, x_max = self.X_train['exam_score_1'].min() - 1, self.X_train['exam_score_1'].max() + 1
        y_min, y_max = self.X_train['exam_score_2'].min() - 1, self.X_train['exam_score_2'].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                            np.arange(y_min, y_max, 0.01))
        
        # Make predictions for each point in the mesh grid
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        # Plot the contour and training examples
        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(self.X_train['exam_score_1'], self.X_train['exam_score_2'], c=self.y_train, edgecolors='k', marker='o')
        plt.title(title)
        plt.xlabel('Exam Score 1')
        plt.ylabel('Exam Score 2')
        plt.show()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Linear Regression')
    parser.add_argument('-d','--dataset_num', type=str, default = "1", choices=["1","2"], help='string indicating datset number. For example, 1 or 2')
    parser.add_argument('-t','--perform_test', action='store_true', help='boolean to indicate inference')
    args = parser.parse_args()
    classifier = MyLogisticRegression(args.dataset_num, args.perform_test)
    acc = classifier.model_predict_linear()
    # classifier.plot_decision_boundary(classifier.model_linear, "Linear Regression Decision Boundary")
    print(f"LINEAR REGRESSION STATS:\n{acc}")
    acc = classifier.model_predict_logistic()
    # classifier.plot_decision_boundary(classifier.model_logistic, "Logistic Regression Decision Boundary")
    print(f"LOGISTIC REGRESSION STATS:\n{acc}")
    