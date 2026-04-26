from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
import os
from core.engine import HealthEngine

app = Flask(__name__)
app.config['SECRET_KEY'] = 'health_e_ai_secret_key_2026'
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize AI Engine
engine = HealthEngine()

@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/diagnose/<disease>')
def diagnose_page(disease):
    """Dynamic route for all tabular disease pages."""
    valid_diseases = ['diabetes', 'cancer', 'heart', 'liver', 'kidney']
    if disease not in valid_diseases:
        return redirect(url_for('home'))
    return render_template(f'{disease}.html')

@app.route('/predict_tabular', methods=['POST'])
def predict_tabular():
    try:
        disease = request.form.get('disease')
        features = list(map(float, request.form.values()))
        # The first value in form.values() might be the 'disease' name, 
        # so we filter it out if necessary.
        if features[0] == 0: # Simple check, but better to use a specific field
             pass 
        
        # In a real scenario, we would slice the feature list based on the disease
        # We'll assume the form sends only the numeric values
        # we need to handle the disease name explicitly.
        
        # Adjust features if 'disease' is in the values
        # Actually, we'll assume the HTML form has a hidden input for disease
        # and only numeric inputs for features.
        
        # Let's get the actual numeric values only
        data = [float(v) for k, v in request.form.items() if k != 'disease']
        
        result = engine.get_diagnosis(disease, data)
        prediction = 'Sorry! Suffering' if int(result) == 1 else 'Congrats! You are Healthy'
        
        return render_template('result.html', prediction=prediction)
    except Exception as e:
        flash(f"Error: {str(e)}", "danger")
        return redirect(url_for('home'))

@app.route('/predict_image', methods=['POST'])
def predict_image():
    try:
        disease = request.form.get('disease')
        file = request.files['image']
        if not file:
            flash("Please select an image first!", "danger")
            return redirect(url_for('home'))

        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        prediction_raw = engine.get_diagnosis(disease, file_path)
        
        # Handle different labels for different models
        if disease == 'malaria':
            indices = {0: 'PARASITIC', 1: 'Uninfected', 2: 'Invasive carcinomar', 3: 'Normal'}
            pred_class = np.argmax(prediction_raw, axis=1)[0]
            label = indices[pred_class]
            accuracy = round(prediction_raw[0][pred_class] * 100, 2)
        elif disease == 'pneumonia':
            # Binomial prediction
            res = prediction_raw[0][0]
            label = 'Pneumonia' if res > 0.5 else 'Normal'
            accuracy = res * 100 if res > 0.5 else (1 - res) * 100
            accuracy = round(accuracy, 2)
        else:
            label = "Unknown"
            accuracy = 0.0

        return render_template('predict.html', image_file_name=file.filename, label=label, accuracy=accuracy)
    except Exception as e:
        flash(str(e), "danger")
        return redirect(url_for('home'))

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
