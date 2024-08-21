# Sales Prediction Model

## Overview

The Sales Prediction Model is a web application designed to forecast sales based on historical data. It uses sales figures from the first two months and a rate value to predict the sales for the 3rd month. The application provides an intuitive interface to input data and receive predictions, making it a valuable tool for sales forecasting.

## How It Works

1. **Historical Data Input:**
   - **Sales for the 1st Month:** Enter the sales figure for the first month.
   - **Sales for the 2nd Month:** Enter the sales figure for the second month.

2. **Rate Selection:**
   - **Rate:** Choose a rate value from a predefined list (e.g., "zero", "one", "two", etc.) that may influence the prediction.

3. **Prediction Calculation:**
   - The model uses the provided sales data and rate to predict the sales for the 3rd month.

### Example

Here’s an example to illustrate:

1. **Input Data:**
   - **Sales 1st Month:** 2000
   - **Sales 2nd Month:** 2500

2. **Rate Selection:**
   - Choose "three" from the dropdown menu.

3. **Output:**
   - The model will predict the sales for the 3rd month based on the input data and rate.

## Tech Stack

- **Python:** For developing the prediction model and backend logic.
- **Flask:** A lightweight web framework used to build the web application.
- **Pandas:** For data manipulation and preprocessing.
- **NumPy:** For numerical operations and handling arrays.
- **Scikit-learn:** For implementing the regression model.
- **HTML/CSS:** For creating and styling the web interface.

<img width="501" alt="image" src="https://github.com/user-attachments/assets/285a7b5e-0291-420c-8918-05c627a4ef94">
