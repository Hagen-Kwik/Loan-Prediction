from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import pickle
import os

app = Flask(__name__)

loaded_model = pickle.load(open('model.pkl', 'rb'))

gender_mapping = {'Male': 1, 'Female': 0}
ever_married_mapping = {'Yes': 1, 'No': 0}
work_type_mapping = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': 3, 'Never_worked': 4}
residence_type_mapping = {'Urban': 1, 'Rural': 0}
smoking_status_mapping = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}

@app.route('/')
def home():
    return render_template('website.html', prediction_text=None)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data_to_predict = {
            'gender': request.form['gender'],
            'age': int(request.form['age']),
            'hypertension': int(request.form['hypertension']),
            'heart_disease': int(request.form['heart_disease']),
            'ever_married': request.form['ever_married'],
            'work_type': request.form['work_type'],
            'Residence_type': request.form['Residence_type'],
            'avg_glucose_level': float(request.form['avg_glucose_level']),
            'bmi': float(request.form['bmi']),
            'smoking_status': request.form['smoking_status']
        }

        data_to_predict_df = pd.DataFrame([data_to_predict])

        
        session['original_data'] = data_to_predict_df.to_dict(orient='records')

        data_to_predict_df['gender'] = data_to_predict_df['gender'].map(gender_mapping)
        data_to_predict_df['ever_married'] = data_to_predict_df['ever_married'].map(ever_married_mapping)
        data_to_predict_df['work_type'] = data_to_predict_df['work_type'].map(work_type_mapping)
        data_to_predict_df['Residence_type'] = data_to_predict_df['Residence_type'].map(residence_type_mapping)
        data_to_predict_df['smoking_status'] = data_to_predict_df['smoking_status'].map(smoking_status_mapping)

        prediction = loaded_model.predict(data_to_predict_df)
        session['data_to_predict_df'] = data_to_predict_df.to_dict(orient='records')
        session['prediction'] = int(prediction[0])

        return render_template('index.html', prediction_text=f'Prediction: {prediction[0]}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')



@app.route('/update_data', methods=['POST'])
def update_data():
    try:
        user_input = request.form.get('prediction')

        if user_input not in ['True', 'False']:
            raise ValueError('Invalid user input.')

        original_data = pd.DataFrame(session.pop('original_data', None))
        prediction = session.pop('prediction', None)

        if original_data is None or prediction is None:
            raise ValueError('Data or prediction not found in session.')
        
        if prediction == 0 and user_input == 'True':
            original_data['stroke'] = 0
        elif prediction == 0 and user_input == 'False':
            original_data['stroke'] = 1
        elif prediction == 1 and user_input == 'True':
            original_data['stroke'] = 1
        elif prediction == 1 and user_input == 'False':
            original_data['stroke'] = 0

        columns_order = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married', 'work_type',
                         'Residence_type', 'avg_glucose_level', 'bmi', 'smoking_status', 'stroke']
        original_data = original_data[columns_order]

        original_data.to_csv('newdata.csv', mode='a', header=not os.path.exists('newdata.csv'), index=False)

        return redirect(url_for('home'))
    except Exception as e:
        return render_template('website.html', prediction_text=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)