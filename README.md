# Final Project: Solving Edutech Company Problems

## Business Understanding

The project aims to assist an edutech company in predicting student dropout risks. High dropout rates are a critical issue for educational institutions, affecting both their reputation and financial stability. By identifying at-risk students early, the company can intervene proactively to improve student retention and success.

### Business Problems

The key business problems addressed in this project include:

- High student dropout rates impacting the institution's performance.
- Lack of early detection mechanisms for identifying students at risk of dropping out.
- Inadequate insights into factors contributing to student attrition.

### Project Scope

The project focuses on the following areas:

- Data preprocessing and feature engineering to clean and prepare the dataset.
- Developing and training a machine learning model to predict student dropout.
- Deploying the model using Streamlit for easy access and usability.
- Creating a business dashboard with Looker Studio to visualize insights and track student performance.

### Preparation

Data source: [Student Performance Dataset](https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md)

Setup environment:

```bash
# Install necessary packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

## Business Dashboard

The business dashboard, built using Looker Studio, visualizes key insights about student performance and dropout risks. It provides stakeholders with actionable information to identify at-risk students and understand the factors contributing to dropout. You can view the business dashboard here: [Looker Studio Dashboard](https://lookerstudio.google.com/reporting/5bde8ef5-73d2-4edb-8dad-1425eb49d6b2).

## Running the Machine Learning System

To run the machine learning prototype, follow these steps:

1. Clone the repository.
2. Install the required dependencies using `pip install -r requirements.txt`.
3. Run the Streamlit app using `streamlit run app.py`.
4. The app will allow you to input student data and predict the likelihood of dropout.

```bash
# Clone the repository
git clone <repository-url>

# Install necessary packages
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

You can also access the deployed Streamlit application here: [Student Dropout Prediction App](https://student-dropout-prediction-alessandroryo.streamlit.app/).

## Conclusion

The project successfully developed a machine learning model to predict student dropout risks with reasonable accuracy. The model, when integrated with the business dashboard in Looker Studio, provides valuable insights that can help the institution reduce dropout rates and improve overall student retention.

### Recommended Action Items

Here are a few recommended actions for the company to address the dropout issue:

- **Enhance Support for Academic Performance**: Focus on programs that help students complete their curricular units, especially in the first semester, through tutoring, academic counseling, and additional learning resources.
- **Ensure Timely Payment of Tuition Fees**: Introduce financial aid or flexible payment options to help students manage their tuition fees and reduce the risk of dropout due to financial difficulties.
- **Address Pre-existing Academic Gaps**: Provide training or development programs for students with lower previous qualification grades to ensure they are prepared for the higher academic demands of college.
- **Increase Monitoring for Debtor Students**: Enhance monitoring and support for students with financial debt to reduce stress and ensure they remain focused on their studies.
- **Expand Scholarship Programs**: Consider expanding scholarship programs to support more students, particularly those identified as being at higher risk of dropout.