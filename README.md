# AIProject


## https://aiproject-nl43.onrender.com   
click to visit


# Heart Disease Prediction Web Application

## Overview
The Heart Disease Prediction Web Application is a Flask-based web application that predicts the likelihood of a person having heart disease based on their medical and physical parameters. The application uses a pre-trained machine learning model to analyze user inputs and return a prediction.

## Features
- **User-Friendly Form**: A responsive and visually appealing form for users to input their details.
- **Machine Learning Integration**: Predicts the likelihood of heart disease using a Logistic Regression model.
- **Dynamic Results Display**: Displays the prediction results on the same page after form submission.
- **Lightweight and Fast**: Built with Flask for seamless performance.
- **Responsive Design**: Fully functional on desktop and mobile devices.

## Project Structure
```
project/
├── static/
│   ├── css/
│   │   └── styles.css  # Custom CSS for styling
├── templates/
│   ├── index.html       # Main HTML template
├── app.py               # Main application file
├── model.pkl            # Pre-trained machine learning model
├── requirements.txt     # Dependencies for the project
└── README.md            # Project documentation
```

## Technologies Used
- **Frontend**:
  - HTML5
  - CSS3
  - Bootstrap (for additional styling, if used)

- **Backend**:
  - Flask (Python)

- **Machine Learning**:
  - Logistic Regression (Scikit-learn)
  - Libraries: `numpy`, `scikit-learn`, `pandas`

## Installation and Setup
Follow these steps to run the application locally or deploy it to a platform like Render.

### Prerequisites
Ensure you have Python 3.8 or above installed on your system.

### Steps to Run Locally
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/heart-disease-prediction.git
   cd heart-disease-prediction
   ```

2. **Install Dependencies**:
   Install the required Python libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:
   Start the Flask server:
   ```bash
   python app.py
   ```
   The application will be accessible at `http://127.0.0.1:5000/`.

### Deployment on Render
1. **Create a Render Account**:
   - Sign up or log in to [Render](https://render.com).

2. **Set Up the Repository**:
   - Push your code to a GitHub repository.

3. **Deploy**:
   - Create a new web service on Render, connect your repository, and use the following build and start commands:
     - Build Command: `pip install -r requirements.txt`
     - Start Command: `gunicorn app:app`

4. **Access the App**:
   - Once deployed, Render will provide a live URL for your application.

## Usage
1. Open the application in your browser.
2. Fill in all the required fields in the form (e.g., age, sex, cholesterol levels, etc.).
3. Click the "Predict" button.
4. View the prediction result displayed below the form.

## Example Input and Output
### Input:
- Age: 45
- Sex: 1 (Male)
- Chest Pain Type: 2
- Resting Blood Pressure: 120
- Cholesterol: 240

### Output:
- "The patient is likely to have heart disease."

## License
This project is open-source and available under the [MIT License](LICENSE).

## Contact
For any questions or feedback, please reach out to:
- **Name**: Ashutosh Acharya
- **Email**: ashutoshacharya908@gmail.com


