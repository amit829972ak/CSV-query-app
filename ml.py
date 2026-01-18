import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import time
import requests
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from io import StringIO
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier,GradientBoostingRegressor
from xgboost import XGBClassifier,XGBRegressor
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_auc_score, r2_score,mean_squared_error, silhouette_score
import streamlit.components.v1 as components
import chardet

# Animation at start
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_url_hello = "https://lottie.host/c7ae530f-5d35-4fe1-851e-12bc2b7864f4/2rcYkXCxHr.json"
lottie_hello = load_lottieurl(lottie_url_hello)

st_lottie(lottie_hello)

def main():
    st.header("Supervised Machine Learning")

    st.subheader("Upload your CSV file")
    data_file = st.file_uploader("Upload your CSV file", type=["csv"])
    st.write("---")
    

    if data_file is not None:
        if data_file.name.endswith(".csv"):
           result = chardet.detect(data_file.read())
           encoding = result['encoding']
           data_file.seek(0)
           data = pd.read_csv(data_file,encoding=encoding)
        st.write(data.head())   
        st.write('---')
        st.subheader('We can see the count value, mean value, standard deviation, minimum and maximum value of each numeric column.')
        st.write(data.describe())
        st.write("---")        
        
        
        st.subheader('We can see the data type, null values and unique value for each column.')
        result = {'Column': [],'Data Type': [],'Null Values': [],'Unique Values': []}

# Populate the dictionary with the results
        for col in data.columns:
            result['Column'].append(col)
            result['Data Type'].append(data[col].dtype)
            result['Null Values'].append(data[col].isnull().sum())
            result['Unique Values'].append(data[col].nunique())
            

# Convert the dictionary to a DataFrame
        result = pd.DataFrame(result)

# Show the table
        st.write(result)
        # Visualization      
        st.sidebar.header("Visualizations")
        plot_options = ["None","Bar plot", "Scatter plot", "Histogram", "Box plot", "Count plot"]
        selected_plot = st.sidebar.selectbox("Choose a plot type", plot_options)

        if selected_plot == "Bar plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Bar plot:")
            fig, ax = plt.subplots()
            sns.barplot(x=data[x_axis], y=data[y_axis], ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right") 
            st.pyplot(fig)

        elif selected_plot == "Scatter plot":
            x_axis = st.sidebar.selectbox("Select x-axis", data.columns)
            y_axis = st.sidebar.selectbox("Select y-axis", data.columns)
            st.write("Scatter plot:")
            fig, ax = plt.subplots()
            sns.scatterplot(x=data[x_axis], y=data[y_axis], ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Histogram":
            column = st.sidebar.selectbox("Select a column", data.columns)
            bins = st.sidebar.slider("Number of bins", 5, 100, 20)
            st.write("Histogram:")
            fig, ax = plt.subplots()
            sns.histplot(data[column], bins=bins, ax=ax)
            st.pyplot(fig)

        elif selected_plot == "Box plot":
            column = st.sidebar.multiselect("Select a column", data.columns)
            st.write("Box plot:")
            fig, ax = plt.subplots()
            sns.boxplot(data[column], ax=ax)
            st.pyplot(fig)
         
        elif selected_plot == "Count plot":
            column = st.sidebar.selectbox("Select a column", data.columns)
            hue = st.sidebar.selectbox("Select hue (optional)", ["None"] + list(data.columns))
            if hue == "None":
               hue = None
            st.write("Count plot:")
            fig, ax = plt.subplots()
            sns.countplot(x=column, hue=hue, data=data, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            st.pyplot(fig)
            
                
        # filling the null values
        for col in data.columns:
            if data[col].dtype == 'object':
                if not data[col].mode().empty:
                    data[col] = data[col].fillna(data[col].mode().iloc[0])
            elif data[col].dtype == 'float':
               data[col] = data[col].fillna(data[col].mean())
            elif data[col].dtype == 'int':
               data[col] = data[col].fillna(data[col].median())
        # showing the outliers percentage
        st.write('---')
        outlier_percentages = {}
        for col in data.columns:
            if np.issubdtype(data[col].dtype, np.number):
                q1 = data[col].quantile(0.25)
                q3 = data[col].quantile(0.75)
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                outlier_percentage = len(outliers) / len(data) * 100
                outlier_percentages[col] = outlier_percentage

        st.subheader('We can see the percentage of outliers:')
        for col, percentage in outlier_percentages.items():
            
            st.write(f"{col}: {percentage:.2f}%")
        st.write('---')         

        result = {'Column': [],'Data Type': [],'Null Values': [],'Unique Values': []}

# Populate the dictionary with the results
        for col in data.columns:
            result['Column'].append(col)
            result['Data Type'].append(data[col].dtype)
            result['Null Values'].append(data[col].isnull().sum())
            result['Unique Values'].append(data[col].nunique())
            

# Convert the dictionary to a DataFrame
        result = pd.DataFrame(result)

# Show the table
        st.subheader('We have removed all the null values.')
        st.write(result)
        st.write(data)
        st.write('---')
    
    # Remove outliers from the selected columns using the median method            
        columns = st.multiselect('Choose columns to remove outliers from', data.columns)
         
        data_clean = data.copy()
        for column in columns:
        # Calculate the median and interquartile range (IQR)
            Q1 = data[column].quantile(0.25)
            Q3 = data[column].quantile(0.75)
            IQR = Q3 - Q1

        # Remove outliers
            data_clean = data_clean[~((data_clean[column] < (Q1 - 1.5 * IQR)) | (data_clean[column] > (Q3 + 1.5 * IQR)))]

    # Show the first 5 rows of the cleaned data
        st.write('Cleaned Data')
        st.write(data_clean.head())
                    
        outlier_percent = {}
        for col in data_clean.columns:
                if np.issubdtype(data_clean[col].dtype, np.number):
                   q1 = data_clean[col].quantile(0.25)
                   q3 = data_clean[col].quantile(0.75)
                   iqr = q3 - q1
                   lower_bound = q1 - 1.5 * iqr
                   upper_bound = q3 + 1.5 * iqr
                   outliers = data_clean[(data_clean[col] < lower_bound) | (data_clean[col] > upper_bound)]
                   outlier_per = len(outliers) / len(data_clean) * 100
                   outlier_percent[col] = outlier_per

# Display the percentage of outliers for each column
        st.subheader('We have reduced the outliers:')
        for col, percentage in outlier_percent.items():
            st.write(f"{col}: {percentage:.2f}%")
        st.write('---')
        
        # Apply ordinal encoding to the selected columns
        ordinal_encoders = {}       
        columns = st.multiselect('Select columns for encoding', data_clean.columns)

        if len(columns) > 0:
            encoder = OrdinalEncoder()
            data_clean[columns] = encoder.fit_transform(data_clean[columns])
            ordinal_encoders[col] = encoder
# Display the updated DataFrame
        st.write(data_clean.head())
        
        
        
        data_1=data_clean.copy()
        
        # Perform frequency encoding on the selected columns       
        obj_cols = data_1.select_dtypes(include=object)

        for col in obj_cols:
        # Compute the frequency of each value in the column
            freq = data_1[col].value_counts() / len(data_1)
    
        # Map the frequencies to the column
            data_1[col] = data_1[col].map(freq)
            #st.write(data_1.head()) 
        st.write('---')
       
        
        
        # heatmap and corr graph
        st.subheader('Correlation and heatmap')
        fig, ax = plt.subplots(figsize=(10, 8))  # Adjusted for more readable display in Streamlit
        sns.heatmap(data_1.corr(), annot=True, cmap="RdYlGn", annot_kws={"size":8}, ax=ax)
        st.pyplot(fig)
        st.write('---')
         
        #now comes the machine learning 
        # select the column to drop  
        st.header('Model building:')         
        cols_to_drop = st.multiselect("Select columns to drop", data_clean.columns)
        data_clean = data_clean.drop(cols_to_drop, axis=1)
        st.write(data_clean.head())            
                  
        # select the column x and y   
        y_col = st.selectbox("Select y column", data_clean.columns)
        y = data_clean[y_col]
        x = data_clean.drop(y_col, axis=1)
        st.write("Column of features : X", x)
        st.write("Target Column : Y", y.head())    
        st.write('---')
        
        st.subheader('Scaling:')      
        scale = StandardScaler()
        scaled = scale.fit_transform(x)
        
        
        x_scaled=pd.DataFrame(data=scaled,columns=x.columns)
        st.write("Scaled :",x_scaled)
        st.write('---')
        
        # smoting of x and y
        # Create a checkbox to allow the user to choose whether or not to use SMOTE
        st.subheader('Data Splitting and SMOTE')
        st.write("Target variable distribution:")
        st.write(y.value_counts())
        
        use_smote = st.checkbox('Use SMOTE to oversample the minority class')

        # Create a slider to allow the user to choose the test size
        test_size = st.slider('Test size', 0.1, 0.9, 0.3)

        if use_smote:
            # Use SMOTE to oversample the minority class in the data
            sm = SMOTE(random_state=42)
            X_resampled, y_resampled = sm.fit_resample(x_scaled, y)
            st.write("After SMOTE - Target variable distribution:")
            st.write(y_resampled.value_counts())
            # Split the resampled data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=test_size, random_state=42)
        else:
            # Split the original data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(x_scaled, y, test_size=test_size, random_state=42)

        # Display the shapes of the resulting arrays
        st.write(f"X_train shape: {X_train.shape}")
        st.write(f"X_test shape: {X_test.shape}")
        st.write(f"y_train shape: {y_train.shape}")
        st.write(f"y_test shape: {y_test.shape}")
        st.write('---')
        
# Model selection
        st.header('Model Training and Evaluation')        
# Create a dictionary of classifiers
        task = st.selectbox('Select machine learning task', ['None','Classification', 'Regression', 'Clustering'])
        if task == 'Classification':
            algorithm = st.selectbox('Select algorithm', ['Logistic Regression', 'XGBoost', 'Decision Tree', 'Random Forest', 'SVM', 'KNN', 'Gradient Boosting'])
            if algorithm == 'Logistic Regression':
               model = LogisticRegression(max_iter=1000)
            elif algorithm == 'XGBoost':
               model = XGBClassifier(eval_metric='logloss')
            elif algorithm == 'Decision Tree':
               model = DecisionTreeClassifier()
            elif algorithm == 'Random Forest':
               model = RandomForestClassifier()
            elif algorithm == 'SVM':
               model = SVC(probability=True)
            elif algorithm == 'KNN':
               model = KNeighborsClassifier()
            elif algorithm == 'Gradient Boosting':
               model = GradientBoostingClassifier()
            
            with st.spinner('Training model...'):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            st.success('Model training completed!')
            st.subheader('Classification Results')
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.write(report_df)
            st.write('F1 score:', f1_score(y_test, y_pred,average='weighted'))
            st.write('Precision score:', precision_score(y_test, y_pred,average='weighted'))
            st.write('Recall score:', recall_score(y_test, y_pred,average='weighted'))
            st.write('Accuracy score:', accuracy_score(y_test, y_pred))
            st.write('Confusion matrix:')
            st.write(confusion_matrix(y_test, y_pred))
            if len(set(y)) > 2:
                y_prob = model.predict_proba(X_test) 
                st.write('ROC AUC score (one-vs-rest):', roc_auc_score(y_test, y_prob, multi_class='ovr'))
            else:
                st.write('ROC AUC score:', roc_auc_score(y_test, y_pred))
            
            # PREDICTOR FEATURE FOR CLASSIFICATION
            st.write('---')
            st.header('🔮 Make Predictions')
            st.subheader('Enter feature values for prediction')
            
            input_data = {}
            cols = st.columns(2)
            for idx, col_name in enumerate(x.columns):
                with cols[idx % 2]:
                    input_data[col_name] = st.number_input(f'{col_name}', value=float(x[col_name].mean()))
            
            if st.button('Predict', type='primary'):
                # Create input dataframe
                input_df = pd.DataFrame([input_data])
                
                # Scale the input
                input_scaled = scale.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                prediction_proba = model.predict_proba(input_scaled)
                
                st.success(f'**Predicted Class: {prediction[0]}**')
                
                # Show probabilities
                st.write('**Class Probabilities:**')
                proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
                st.dataframe(proba_df.style.highlight_max(axis=1))
                
        elif task == 'Regression':
            algorithm = st.selectbox('Select algorithm', ['Linear Regression', 'XGBoost', 'Decision Tree', 'Random Forest', 'KNN', 'Gradient Boosting'])
            if algorithm == 'Linear Regression':
                model = LinearRegression()
            elif algorithm == 'XGBoost':
                model = XGBRegressor(eval_metric='rmse')
            elif algorithm == 'Decision Tree':
                model = DecisionTreeRegressor()
            elif algorithm == 'Random Forest':
                model = RandomForestRegressor()
            elif algorithm == 'KNN':
                model = KNeighborsRegressor()
            elif algorithm == 'Gradient Boosting':
                model = GradientBoostingRegressor()
            
            with st.spinner('Training model...'):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            st.success('Model training completed!')
            st.subheader('Regression Results')
            st.write('R2 score:', r2_score(y_test, y_pred))
            st.write('Mean squared error:', mean_squared_error(y_test, y_pred))
            st.write('Root mean squared error:', np.sqrt(mean_squared_error(y_test, y_pred)))
            
            # PREDICTOR FEATURE FOR REGRESSION
            st.write('---')
            st.header('🔮 Make Predictions')
            st.subheader('Enter feature values for prediction')
            
            input_data = {}
            cols = st.columns(2)
            for idx, col_name in enumerate(x.columns):
                with cols[idx % 2]:
                    input_data[col_name] = st.number_input(f'{col_name}', value=float(x[col_name].mean()))
            
            if st.button('Predict', type='primary'):
                # Create input dataframe
                input_df = pd.DataFrame([input_data])
                
                # Scale the input
                input_scaled = scale.transform(input_df)
                
                # Make prediction
                prediction = model.predict(input_scaled)
                
                st.success(f'**Predicted Value: {prediction[0]:.4f}**')
                
                # Show comparison with actual range
                st.write(f'**Target variable range in dataset:**')
                st.write(f'Min: {y.min():.4f}, Max: {y.max():.4f}, Mean: {y.mean():.4f}')
            
        elif task == 'Clustering':
            algorithm = st.selectbox('Select algorithm', ['K-means'])
            if algorithm == 'K-means':
                n_clusters = st.slider('Select number of clusters', 2, 10)
                model = KMeans(n_clusters=n_clusters, random_state=42)
                
                with st.spinner('Training clustering model...'):
                    model.fit(data_clean)
                    labels = model.labels_
                
                st.success('Clustering completed!')
                silhouette_avg = silhouette_score(data_clean, labels)
                st.write('Silhouette score:', silhouette_avg)
                
                # Elbow method
                st.subheader('Elbow Method for Optimal K')
                distortions = []
                for i in range(2, 11):
                    kmeans = KMeans(n_clusters=i, random_state=42)
                    kmeans.fit(data_clean)
                    distortions.append(kmeans.inertia_)
                
                fig, ax = plt.subplots()
                ax.plot(range(2, 11), distortions, 'bx-')
                ax.set_xlabel('Number of clusters')
                ax.set_ylabel('Distortion')
                ax.set_title('Elbow Method')
                st.pyplot(fig)
                
                data_clean['cluster'] = labels
                st.write(data_clean)
                
                # PREDICTOR FEATURE FOR CLUSTERING
                st.write('---')
                st.header('🔮 Predict Cluster')
                st.subheader('Enter feature values to predict cluster assignment')
                
                input_data = {}
                cols = st.columns(2)
                feature_cols = [col for col in data_clean.columns if col != 'cluster']
                for idx, col_name in enumerate(feature_cols):
                    with cols[idx % 2]:
                        input_data[col_name] = st.number_input(f'{col_name}', value=float(data_clean[col_name].mean()))
                
                if st.button('Predict Cluster', type='primary'):
                    # Create input dataframe
                    input_df = pd.DataFrame([input_data])
                    
                    # Predict cluster
                    cluster = model.predict(input_df)
                    
                    st.success(f'**Predicted Cluster: {cluster[0]}**')
                    
                    # Show cluster statistics
                    st.write(f'**Cluster {cluster[0]} Statistics:**')
                    cluster_data = data_clean[data_clean['cluster'] == cluster[0]]
                    st.write(cluster_data.describe())
                
if __name__ == "__main__":
    main()
