from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Sample data for the model
data = {
    'Rate': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sales_1st_Month': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    'Sales_2nd_Month': [110, 160, 210, 260, 310, 360, 410, 460, 510, 560],
    'Sales_3rd_Month': [120, 170, 220, 270, 320, 370, 420, 470, 520, 570]
}

df = pd.DataFrame(data)

# Preprocessing
df['Rate'] = df['Rate'].astype(int)
X = df[['Rate', 'Sales_1st_Month', 'Sales_2nd_Month']]
y = df['Sales_3rd_Month']

# Train the model
model = LinearRegression()
model.fit(X, y)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            rate = request.form['rate']
            sales_1st_month = float(request.form['sales_1st_month'])
            sales_2nd_month = float(request.form['sales_2nd_month'])
            
            # Convert rate to integer
            rate_dict = {
                "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
                "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
            }
            rate_int = rate_dict.get(rate, 0)
            
            # Prepare the input for prediction
            X_new = pd.DataFrame([[rate_int, sales_1st_month, sales_2nd_month]],
                                 columns=['Rate', 'Sales_1st_Month', 'Sales_2nd_Month'])
            
            # Predict
            prediction = model.predict(X_new)[0]
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
