# Web-Based Application with Machine Learning Model: Car Price Prediction

This repository contains a web-based application that integrates a machine learning model using Dash by Plotly. The project focuses on creating an aesthetically appealing user interface while maintaining robust functionality. It also includes mechanisms for automatic imputation of form fields if they are skipped by users.

## Features

- **Machine Learning Integration**: A pre-trained machine learning model is incorporated into the application for predictions.
- **Dash by Plotly**: The application is built using Dash, providing interactive visualizations and seamless UI integration.
- **Form Imputation**: If users skip any fields in the form, the application automatically fills them using an imputation technique.
- **Responsive Design**: The layout is optimized for various screen sizes, ensuring accessibility on mobile and desktop devices.
- **Data Transformation**: Includes Min-Max scaling and imputation techniques for handling missing data.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AIT-MachineLearning/a3-predicting-car-prices-sachinmalego.git

2. Navigate into the project directory:
   ```bash
   cd <yourproject_dir>

3. Run Dockerfile:
   ```bash
   docker compose up

4. Run the application:
   ```bash
    python3 main.py

## Project Structure

- `main.py`: Main application file.
- `pages/`: Folder containing pages UI.
- `model/`: Folder containing the machine learning model.
- `LinearRegression.py`: Script for handling classes.
- `.Dockerfile`: Installs list of dependencies needed to run the application.
- `docker-compose.yaml`: Needed to run the dockerize the container.
- `docker-compose-deploy.yaml`: Needed for deployment to the server.
- `.devcontainer/source_code`: Consists of all the .ipynb and .py files.
- `st125171_sachin_A2_predictcarpricetwo_orgi_dev_vone.pdf`: Report file of the assignment.
- `README.md`: Project documentation.

## Usage

Once the application is running, users can interact with the web interface to input data. If certain fields are left blank, the system automatically imputes missing values and provides predictions using the integrated machine learning model.

### Key Functionalities:

1. **Prediction Model**: Upload a dataset, and the system generates predictions based on the machine learning model.
2. **Imputation**: Form fields that are left blank are filled using an imputation technique to ensure data completeness.

## Contributing

Contributions are welcome! Please submit a pull request or raise an issue for any suggestions or improvements.

## License

This project is open source for all to use. Enjoy!!!

## RUNNING APPLICATION:
mlflow logs: https://mlflow.ml.brain.cs.ait.ac.th/   
docker hub image: https://hub.docker.com/repositories/sachinmalego  
brain lab server: st125171_4.ml.brain.cs.ait.ac.th/predictvthree
