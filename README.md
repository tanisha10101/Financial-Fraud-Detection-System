# Financial Fraud Detection Model

## Overview  
This project focuses on detecting fraudulent transactions in financial systems. The analysis is centered on two specific transaction types: **TRANSFER** and **CASH_OUT**, with fraudulent transactions involving only customer accounts. For fraudulent **TRANSFER** transactions, the recipient account has been used 4097 times for valid cash withdrawals, highlighting potential patterns of misuse.  

Multiple Machine Learning (ML) and Neural Network (NN) models were implemented and evaluated to achieve high accuracy in fraud detection.

## Models and Performance  

### Machine Learning Models  
Four ML models were used for prediction, with their respective performance metrics:  
1. **Logistic Regression**  
   - **AUC**: 0.983745  
   - **Hamming Loss**: 0.000834  
   - Best-performing ML model.  

2. **Random Forest**  
   - **AUC**: 0.746883  
   - **Hamming Loss**: 0.001035  

3. **Extreme Gradient Boosting (XGBoost)**  
   - **AUC**: 0.91755  
   - **Hamming Loss**: 0.001152  

4. **Light Gradient Boosting Machine (LightGBM)**  
   - **AUC**: 0.493639  
   - **Hamming Loss**: 0.00142  
   - Worst-performing ML model.  

### Neural Networks  
Two NN architectures were tested, showing better overall performance than ML models:  
1. **Deep Neural Network (DNN)**  
   - **AUC**: 0.995656  
   - **Hamming Loss**: 0.000538  

2. **Recurrent Neural Network (RNN)**  
   - **AUC**: 0.997087  
   - **Hamming Loss**: 0.000471  
   - Best-performing model overall.  

## Key Insights  
- **Neural Networks outperform Machine Learning models**, as expected, with RNN being the top performer.  
- Logistic Regression is the best among the ML models, while LightGBM performs the worst.  
- Fraud detection is optimized for transactions involving **TRANSFER** and **CASH_OUT** types, focusing solely on customer accounts.  

## Repository Contents  
- **`financial_fraud_detection.ipynb`**: The Jupyter Notebook containing all code and analysis for model implementation, training, and evaluation.  
- **Data**: Dataset used for training and evaluation (link or details if applicable).  
- **Results**: Comparative analysis of all models with metrics.  

## Getting Started  

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-username/financial-fraud-detection.git
   cd financial-fraud-detection
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Jupyter Notebook to explore the implementation:
   ```bash
   jupyter notebook financial_fraud_detection.ipynb

## Dependencies
- Python 3.x
- **Libraries**: scikit-learn, XGBoost, LightGBM, TensorFlow/PyTorch, Pandas, NumPy, Matplotlib, etc.

## Future Work
- Integration with live transaction systems for real-time fraud detection.
- Exploring additional neural network architectures like LSTMs and GRUs.
- Testing on larger, real-world datasets to improve robustness.

## License  
This project is licensed under the [MIT License](LICENSE).  
