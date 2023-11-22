from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import io
import base64


# obter os nomes dos classificadores disponíveis
def classificadores(name):
    classifiers = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(),
        'MLP': MLPClassifier(),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier()
    }
    return classifiers.get(name)


# hiperparâmetros e seus tipos para o classificador
def parametros(name):
    parameters = {
        'KNN': {'n_neighbors': {'description': 'Number of neighbors', 'type': int}},
        'SVM': {'C': {'description': 'Regularization parameter', 'type': float}},
        'MLP': {'hidden_layer_sizes': {'description': 'Number of neurons in the hidden layer', 'type': list}},
        'DT': {'max_depth': {'description': 'Maximum depth of the tree', 'type': int}},
        'RF': {'n_estimators': {'description': 'Number of trees in the forest', 'type': int}}
    }
    return parameters.get(name)


# gerar a imagem da matriz de confusão
def matrix_confusao(y_true, y_pred):
    confusion_matrix = metrics.confusion_matrix(y_true, y_pred)

    # plotando a matriz de confusão
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap='viridis')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')

    return img_str


# treinar o classificador e avaliar seu desempenho e retornar a matriz de confusão
def treinar(classifier):
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # teste do conjunto de dados
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)

    # calculando as métricas de avaliação
    accuracy = metrics.accuracy_score(y_test, y_pred)
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    # mede a proporção de exemplos positivos que foram corretamente identificados
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    # mede a precisão e o recall em uma única pontuação
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    # matriz de confusão
    plt.figure()
    plt.matshow(confusion_matrix, cmap='Blues')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            plt.text(j, i, str(confusion_matrix[i, j]), ha='center', va='center')

    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png')
    img_buf.seek(0)
    img_str = base64.b64encode(img_buf.read()).decode('utf-8')

    return accuracy, precision, recall, f1_score, img_str


# obter os nomes dos classificadores disponíveis
def get_classifier_names():
    return ['KNN', 'SVM', 'MLP', 'DT', 'RF']

