# 🌾 AI-Powered Crop Recommendation System 🚀  

## 🌱 Overview  

Farming meets artificial intelligence! Our AI-powered crop recommendation system is here to revolutionize agriculture by predicting crop suitability based on environmental conditions and market demand. Leveraging machine learning, this tool provides intelligent insights to help farmers make informed decisions and maximize yield.  

## 🌟 Features  

- **🌾 Smart Crop Suitability Prediction**: Uses a trained RandomForestClassifier model to evaluate crop suitability based on key environmental factors.  
- **🛠️ Interactive User Inputs**: Manually enter environmental conditions or select a crop from the dataset for a guided recommendation.  
- **📊 Intelligent Data Processing**: Automatically encodes suitability labels and prepares the dataset for analysis.  
- **⚙️ Advanced Model Training & Evaluation**: Implements a Random Forest model, trains it with 80%-20% data split, and evaluates performance.  
- **💻 Beautiful & Intuitive UI**: Built with Streamlit for an interactive, user-friendly experience.  

## 🌍 Try the Website  

🔗 **Experience the AI-Powered Crop Recommendation System Now:**  
👉 [Crop Recommendation System](https://thepavan1-crop-recommendation-system.streamlit.app/)  

## 🛠 Installation  

### 📌 Prerequisites  

Ensure you have Python installed and set up the necessary dependencies:  

```sh
pip install pandas scikit-learn streamlit numpy
```  

## 🚀 How to Run  

1. Place the dataset file `crop_data.csv` in the project directory.  
2. Launch the application using:  

```sh
streamlit run app.py
```  

3. The app will open in your web browser, allowing you to input environmental parameters and receive AI-powered crop recommendations.  

## 📂 Dataset  

The dataset (`crop_data.csv`) consists of:  

- **🌿 Crop Name**: The name of the crop.  
- **🌡️ Temperature (°C)**: Recommended temperature range.  
- **💦 Rainfall (mm)**: Optimal rainfall requirement.  
- **🧪 Soil pH**: Ideal soil pH level.  
- **📈 Market Demand (1-10)**: Demand rating on a scale of 1 to 10.  

## 🏗️ Model Training Process  

- 🌍 The suitability of crops is classified using predefined thresholds.  
- 🔄 Categorical labels are encoded with `LabelEncoder`.  
- 📊 The dataset is split into training (80%) and testing (20%) sets.  
- 🌲 A RandomForestClassifier is trained to predict crop suitability.  
- 📉 Accuracy is evaluated using `accuracy_score`.  

## 🎯 Usage Guide  

1. Select a crop from the dropdown to autofill its conditions or manually enter values.  
2. Click **"Recommend Crops"** to get AI-generated suitability predictions.  
3. View model accuracy and dataset insights directly in the app.  

## 🔮 Future Enhancements  

- 🌦 **Real-time Weather API Integration**: Incorporate real-world weather conditions for more precise recommendations.  
- 🧠 **Advanced AI Models**: Enhance accuracy with deep learning models.  
- 📌 **Expanded Dataset**: Include additional soil nutrients and climate factors for even smarter predictions.  

## 🌏 Impact  

This AI-powered solution empowers farmers with data-driven insights, reducing guesswork and promoting sustainable agriculture. 🌾💡  

🚀 **Let’s grow smarter, together!**  
