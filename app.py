from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('random_forest_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('website.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_data = {
            'loan_amnt': request.form['loan_amnt'],
            'term': int(request.form['term']),
            'int_rate': float(request.form['int_rate']),
            'installment': float(request.form['installment']),
            'emp_length': int(request.form['emp_length']),
            'home_ownership': int(request.form['home_ownership']),
            'annual_inc': float(request.form['annual_inc']),
            'verification_status': int(request.form['verification_status']),
            'dti': float(request.form['dti']),
            'open_acc': int(request.form['open_acc']),
            'pub_rec': int(request.form['pub_rec']),
            'revol_bal': float(request.form['revol_bal']),
            'revol_util': float(request.form['revol_util']),
            'total_acc': int(request.form['total_acc']),
            'application_type': int(request.form['application_type']),
            'mort_acc': int(request.form['mort_acc']),
            'pub_rec_bankruptcies': int(request.form['pub_rec_bankruptcies']),
        }

        data_changed = np.array(list(user_data.values())).reshape(1, -1)
        prediction = model.predict(data_changed)
        print(prediction[0])

        return render_template('predicted.html', data=prediction[0])
    except Exception as e:
        print("Error: {str(e)}")
        return render_template('website.html', prediction_text=f'Error: {str(e)}')



if __name__ == '__main__':
    app.run(debug=True)