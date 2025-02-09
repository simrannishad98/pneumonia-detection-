from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import efficientnet.tfkeras
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model_path = ("C:\\Users\\udayj\\OneDrive\\Desktop\\Final Year Project\\final website\\uday web\\model\\PAI_model_T.h5") # Update the model path accordingly
model = tf.keras.models.load_model(model_path)

@app.route('/')
def ticket():
    return render_template('index.html')


@app.route('/pneumonia/predict', methods=['POST', 'GET']) 
def classify():
    try:
        # Create 'uploads' directory if it doesn't exist
        if not os.path.exists('uploads'):
            os.makedirs('uploads')

        # Save the uploaded file in the 'uploads' directory
        file = request.files['file']
        filepath = os.path.join('uploads', file.filename)
        file.save(filepath)

        # Use the model to make predictions
        preds = pneumoniamodel_predict(filepath, model)

        # Map the predicted probability to the class label and calculate accuracy
        if preds > 0.5:
            predicted_class = "Normal"
            color_class = "green"
            accuracy = preds
        else:
            predicted_class = "Pneumonia"
            color_class = "red"
            accuracy = 1 - preds

        # Format accuracy as percentage with four decimal places
        accuracy_percentage = "{:.4f}%".format(accuracy * 100)

        # Prepare the response with prediction result, accuracy, and CSS class
        prediction_response = f'<span class="{color_class}">{predicted_class}</span>'
        accuracy_response = f'{accuracy_percentage}'

        return jsonify({'result': prediction_response, 'accuracy': accuracy_response})

    except Exception as e:
        return jsonify({'error': str(e)})


def pneumoniamodel_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0 

    preds = model.predict(x)
    return preds[0][0]  # Return the probability of pneumonia

if __name__ == '__main__':
    app.run(debug=True)
