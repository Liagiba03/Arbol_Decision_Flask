from flask import Flask, render_template, url_for, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, recall_score, f1_score, precision_score
from pandas import DataFrame
from graphviz import Source
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import os
import requests

app = Flask(__name__)

# Función para descargar archivos desde Google Drive
def download_file_from_drive(url, dest_path):
    if not os.path.exists(dest_path):  # Solo descargar si el archivo no existe
        print(f"Descargando {dest_path}...")
        response = requests.get(url, stream=True)
        with open(dest_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
        print(f"Archivo guardado en {dest_path}")
    else:
        print(f"{dest_path} ya existe. No se descargó.")


# Construcción de una función que realice al particionado conmpleto
def train_val_test_split(df, rstate = 42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state = rstate, shuffle = shuffle, stratify= strat)
    strat = train_test[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size = 0.5, random_state = rstate, shuffle= shuffle, stratify= strat)
    return (train_set, val_set, test_set)

def remove_labels(df, label_name):
    X = df.drop(label_name, axis = 1)
    y = df[label_name].copy()
    return (X, y)

def evaluate_result(y_pred, y, y_prep_pred, y_prep, metric):
    print(metric, __name__, "WITHOUT preparation: ", metric(y_pred, y, average='weighted'))
    print(metric, __name__, "WITH preparation: ", metric(y_prep_pred, y_pred, average = 'weighted'))

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate-tree', methods=['GET'])
def generate_tree():

    df = pd.read_csv("datasets/TotalFeatures-ISCXFlowMeter.csv")

    # Copiar el DataSet y transformar a varible de salida a numérica
    # para calcular las correlaciones
    X = df.copy()
    X['calss'] = X['calss'].factorize()[0]

    # División del DataSet
    train_set, val_set, test_set = train_val_test_split(X)

    X_train, y_train = remove_labels(train_set, 'calss')
    X_val, y_val = remove_labels(val_set, 'calss')
    X_test, y_test = remove_labels(test_set, 'calss')

    # Robust Scaler
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    scaler = RobustScaler()
    X_test_scaled = scaler.fit_transform(X_test)

    scaler = RobustScaler()
    X_val_scaled = scaler.fit_transform(X_val)

    # Realizar la transformación a un DataFrame de Pandas
    X_train_scaled = DataFrame(X_train_scaled, columns = X_train.columns, index = X_train.index)

    # Se reduce el numero de atributos del DataSet para visualizarlo mejor

    X_train_reduced = X_train[['min_flowpktl', 'flow_fin']]

    # Se genera un modelo con el DataSet reducido
    clf_tree_reduced = DecisionTreeClassifier(max_depth=2, random_state = 42)
    clf_tree_reduced.fit(X_train_reduced, y_train)

    # Exportar árbol como gráfico
    dot_path = os.path.join('static', 'android_malware.dot')
    png_path = os.path.join('static', 'tree.png')
    export_graphviz(
        clf_tree_reduced,
        out_file=dot_path,
        feature_names=X_train_reduced.columns,
        class_names=["benign", "adware", "malware"],
        rounded=True,
        filled=True
    )
    # Convertir .dot a .png
    graph = Source.from_file(dot_path)
    graph.format = 'png'
    graph.render(filename='static/tree', cleanup=True)


    # Devolver la ruta de la imagen generada
    return jsonify({'tree_image': url_for('static', filename='tree.png')})

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port= 5000)