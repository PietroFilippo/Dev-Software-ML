from flask import Blueprint, render_template, request, jsonify
from sklearn import metrics, datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from app.classifiers import classificadores as classifiers, parametros as parameters, treinar, get_classifier_names
import io
import base64


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
            return jsonify({'error': f'valor de {parameter_name} deve ser um número inteiro ou float.'}), 400
    elif parameter_name == 'C':
        try:
            parametros[parameter_name] = float(parameter_value)
        except ValueError:
            return jsonify({'error': f'valor de {parameter_name} deve ser um número inteiro ou float.'}), 400
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
