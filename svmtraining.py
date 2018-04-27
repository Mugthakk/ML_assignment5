import numpy as np
import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from preprocessing import *

train_set, test_set = get_train_test_set()


train_features = np.array([train_set[i][0] for i in range(len(train_set))])
test_features = np.array([test_set[i][0] for i in range(len(test_set))])
train_labels = np.array([train_set[i][1] for i in range(len(train_set))])
test_labels = np.array([test_set[i][1] for i in range(len(test_set))])

# Love linear kernel, pca makes no difference , hog
# kernel = 'rbf' will perform badly
svc = SVC(kernel='linear', C=0.85)
svc.fit(X=train_features, y=train_labels)  # train the model
y_prediction_svc_train = svc.predict(train_features)
y_prediction_svc_test = svc.predict(test_features)


def eval_accuracy(true_label, pred_label):
    correct = 0
    for i in range(len(true_label)):
        if true_label[i] == pred_label[i]:
            correct += 1
    return correct/len(true_label)


# TODO plot the data less retarded
def plot_confusion_matrix(cm, title='results', cmap=plt.cm.Reds):
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9','10',
               '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
               '20', '21', '22', '23', '24', '25']
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.1f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')  # "target test data"
    plt.xlabel('Predicted label')
    plt.show()

cnf_matrix = confusion_matrix(test_labels, y_prediction_svc_test)
print(eval_accuracy(test_labels,y_prediction_svc_test))
plot_confusion_matrix(cnf_matrix)
