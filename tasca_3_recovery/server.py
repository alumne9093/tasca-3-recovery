# Importo las bibliotecas necesarias.
import pickle # Para cargar los modelos serializados
from flask import Flask, jsonify, request
import os

# Voy a definir las clases para mis predicciones de púlsares
# Como en el archivo CSV de púlsares la columna target_class es 0 o 1
# entonces asumiré que 0 es No-Pulsar y 1 es Pulsar
pulsar_classes = {
    0: 'No-Pulsar',
    1: 'Pulsar'
}

# Lista de los nombres de las características numéricas en el orden esperado por el scaler y el modelo
# Mantengo el orden como en las columnas para que el Scaler y los modelos reciban las características en el mismo orden
# en que fueron entrenados/ajustados
PULSAR_FEATURES = [
    "mean_integrated_profile",
    "std_integrated_profile",
    "excess_kurtosis_integrated_profile",
    "skewness_integrated_profile",
    "mean_dm_snr_curve",
    "std_dm_snr_curve",
    "excess_kurtosis_dm_snr_curve",
    "skewness_dm_snr_curve"
]

models_dir = 'models'

app = Flask('pulsar-predictor')

# Función de Ayuda para Cargar Modelos y Scaler
def load_model_and_scaler(model_filename):
    file_path = os.path.join(models_dir, model_filename)
    print(f"Intentando cargar modelo desde: {file_path}")
    try:
        with open(file_path, 'rb') as f: # Abro el archivo en modo lectura binaria
            # Deserializo los objetos en el mismo orden que los serialicé
            scaler, model = pickle.load(f)
        print(f"Modelo '{model_filename}' cargado exitosamente")
        return scaler, model
    except FileNotFoundError:
        print(f"Error: Archivo '{file_path}' no encontrado. Asegúrate de que los modelos están en la carpeta '{models_dir}'")
        return None, None
    except Exception as e:
        print(f"Error al cargar el modelo '{model_filename}': {e}")
        return None, None

# Función util para predecir una sola instancia del púlsar
# Le paso los datos de entrada como diccionario, el sc y el modelo entrenado
def predict_single_pulsar(data_request, sc, model):
    # Saco los valores de las características del diccionario de entrada
    features_list = [data_request[feat] for feat in PULSAR_FEATURES]

    # Estandarizo los datos de entrada usando el SC
    data_scaled = sc.transform([features_list])

    # Realizo la predicción usando el modelo cargado que me va a devolver la clase (0 o 1)
    class_pred = model.predict(data_scaled)

    # predict_proba para devolverme las probabilidades para todas las clases
    prob_predicciones = model.predict_proba(data_scaled)

    # Extraigo la probabilidad de la clase positiva
    prob_pulsar = prob_predicciones[0][1]

    # Retorno la clase predicha formateada y la probabilidad de ser púlsar (clase 1)
    return (pulsar_classes[int(class_pred)], float(prob_pulsar))


# Función que maneja la solicitud HTTP POST
# Recibe las instancias del scaler y el modelo cargados.
def handle_pulsar_predict(sc, model):
    # Obtengo los datos del posible púlsar del request body
    data_request = request.get_json()

    # Valido que todas las características esperadas están en el JSON de entrada
    if not all(feat in data_request for feat in PULSAR_FEATURES):
         missing_features = [feat for feat in PULSAR_FEATURES if feat not in data_request]
         return jsonify({"error": "Faltan características en la solicitud", "missing": missing_features}), 400
    # Valido que los valores son numéricos
    if not all(isinstance(data_request[feat], (int, float)) for feat in PULSAR_FEATURES):
         non_numeric_features = [feat for feat in PULSAR_FEATURES if not isinstance(data_request[feat], (int, float))]
         return jsonify({"error": "Algunos valores de características no son numéricos", "non_numeric": non_numeric_features}), 400

    try:
        # Realizo la predicción llamando a la función util
        class_pred, prob_pulsar = predict_single_pulsar(data_request, sc, model)
        
        result = {
            'resultado': class_pred,
            'probabilidad_pulsar': prob_pulsar
        }

        return jsonify(result)

    except KeyError as e:
        return jsonify({"error": f"Característica faltante: {e}"}), 400
    except Exception as e:
        return jsonify({"error": f"Error interno durante la predicción: {e}"}), 500


# --- Rutas de la API para Cada Modelo ---

# Ruta para el modelo de Regresión Logística
@app.route('/predict_lr', methods=['POST'])
def predict_lr_route():
    # Cargo todo
    scaler, lr_model = load_model_and_scaler('lr_pulsar_model.pck')

    # Verifico si la carga fue exitosa
    if not all([scaler, lr_model]):
         # Retorno un error si no se pudo cargar alguno
        return jsonify({"error": "No se pudo cargar el Logistic Regression model"}), 500 # 500 Internal Server Error

    #  Hago la predicción con el modelo y devuelvo el resultaado
    return handle_pulsar_predict(scaler, lr_model)

# Ruta para el modelo Support Vector Machine (SVM)
@app.route('/predict_svm', methods=['POST'])
def predict_svm_route():
    scaler, lr_model = load_model_and_scaler('svm_pulsar_model.pck')

    if not all([scaler, lr_model]):
        return jsonify({"error": "No se pudo cargar el Logistic Regression model"}), 500

    return handle_pulsar_predict(scaler, lr_model)

# Ruta para el modelo Decision Trees
@app.route('/predict_dt', methods=['POST'])
def predict_dt_route():
    scaler, lr_model = load_model_and_scaler('dt_pulsar_model.pck')

    if not all([scaler, lr_model]):
        return jsonify({"error": "No se pudo cargar el Logistic Regression model"}), 500

    return handle_pulsar_predict(scaler, lr_model)

# Ruta para el modelo K Nearest Neighbours
@app.route('/predict_knn', methods=['POST'])
def predict_knn_route():
    scaler, lr_model = load_model_and_scaler('knn_pulsar_model.pck')

    if not all([scaler, lr_model]):
        return jsonify({"error": "No se pudo cargar el Logistic Regression model"}), 500

    return handle_pulsar_predict(scaler, lr_model)

if __name__ == '__main__':
    print("Iniciando servidor Flask...")
    app.run(debug=True, port=8000)
    print("Servidor Flask detenido")