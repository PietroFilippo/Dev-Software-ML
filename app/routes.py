from flask import Blueprint, render_template, request, jsonify
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from app.classifiers import classificadores as classifiers, parametros as parameters
import io
import base64


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
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1_score = metrics.f1_score(y_test, y_pred, average='macro')

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)

    # matriz de confusão
    plt.figure()
    plt.matshow(confusion_matrix, cmap='viridis')
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


main_routes = Blueprint('main', __name__)


# rota para a página inicial
@main_routes.route('/')
def index():
    classifiers = get_classifier_names()
    return render_template('index.html', classifiers=classifiers, confusion_matrix_img=None)


# rota para mostrar os resultados
@main_routes.route('/result', methods=['POST'])
def resultado():
    # obter os dados da solicitação
    classifier_name = request.form['classifier']
    parameter_name = request.form['parameter']
    parameter_value = request.form['value']

    # obtero o classificador e os parâmetros
    classifier = classifiers(classifier_name)
    parametros = parameters(classifier_name)

    # verifica se o parâmetro fornecido é válido para o classificador
    if parameter_name not in parametros:
        return jsonify({'error': f'parâmetro inválido para {classifier_name}'}), 400

        # converter o valor do parâmetro se precisar
    if parameter_name == 'n_neighbors':
        try:
            parametros[parameter_name] = int(parameter_value)
        except ValueError:
            return jsonify({'error': f'valor de {parameter_name} deve ser um número inteiro.'}), 400
    elif parameter_name == 'hidden_layer_sizes':
        try:
            # converter a string para uma lista de inteiros
            sizes = [int(size) for size in parameter_value.split(',')]
            parametros[parameter_name] = sizes
        except ValueError:
            return jsonify(
                {'error': f'valor de {parameter_name} deve ser uma lista de inteiros separados por vírgula.'}), 400
    elif parameter_name in ['max_depth', 'n_estimators']:
        try:
            parametros[parameter_name] = int(float(parameter_value))
        except ValueError:
            return jsonify({'error': f'valor de {parameter_name} deve ser um número inteiro.'}), 400
    elif parameter_name == 'C':
        try:
            parametros[parameter_name] = float(parameter_value)
        except ValueError:
            return jsonify({'error': f'valor de {parameter_name} deve ser um número.'}), 400
    else:
        # caso contrário é usado o valor diretamente
        parametros[parameter_name] = parameter_value

    classifier.set_params(**parametros)

    # treina e avalia o classificador
    accuracy, precision, recall, f1_score, confusion_matrix_img = treinar(classifier)

    # exibe os resultados na página
    return render_template('index.html',
                           classifiers=get_classifier_names(),
                           confusion_matrix_img=confusion_matrix_img,
                           classifier=classifier_name,
                           parameter=parameter_name,
                           value=parameter_value,
                           accuracy=accuracy,
                           precision=precision,
                           recall=recall,
                           f1_score=f1_score)
