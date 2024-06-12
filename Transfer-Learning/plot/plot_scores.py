import matplotlib.pyplot as plt


def Plot_Confusion_Matrix(conf_matrix, class_names=None,cmap=plt.cm.Blues):
    if class_names is None:
        class_names = ['Class1', 'Class2']
    ##
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    fmt = 'd'
    thresh = conf_matrix.max() / 2.
    for i, j in product(range(conf_matrix.shape[0]), range(conf_matrix.shape[1])):
        plt.text(j, i, format(conf_matrix[i, j], fmt),
                horizontalalignment="center",
                color="white" if conf_matrix[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
