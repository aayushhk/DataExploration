import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time
import streamlit.components.v1 as components
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout='wide',page_icon="🍇", page_title="Data Exploration")


models = {
            "Logistic Regression": LogisticRegression(),
            "k-Nearest Neighbors": KNeighborsClassifier(),
            "Support Vector Machine": SVC(probability=True),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
        }

def predict_with_best_model(input_data):
    try:
        
        input_data = np.array(input_data).reshape(1, -1)
        prediction = best_model.predict(input_data)
        return prediction[0]
    except Exception as e:
        return None

# Initialize session state to control reloading and task flow
if 'data_loaded' not in st.session_state:
    st.session_state['data_loaded'] = False
if 'features_selected' not in st.session_state:
    st.session_state['features_selected'] = False
if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False

# Streamlit app title and description


st.title("Classification Model Comparison App")
st.write("Upload a dataset to apply and compare different classification models.")

a=st.container(border=True)
# Upload dataset
def plot_advanced_visualizations(data):
    
    numeric_columns = data.select_dtypes(include=['float64', 'int64','character']).columns
    snumeric_columns = data.columns
    
    if len(numeric_columns) < 2:
        st.warning("The dataset must have at least two numeric columns for visualization.")
        return

    # Dropdown for visualization type
    graph_type = st.selectbox("Select Visualization Type", 
                              ["Correlation Heatmap", "Distribution Plot", "Pairplot", 
                               "Boxplot", "Violin Plot", "2D Scatter Plot", 
                               "3D Scatter Plot", "Line Plot", "Density Heatmap", 
                               "Bubble Chart", "Hexbin Plot", "Scatter Matrix"])
    if graph_type:
        st.session_state['graph_type'] = graph_type
    # Select X, Y, and Z axes based on graph requirements
    c1,c2,c3,c4=st.columns(4)
    with c1:

        x_axis = st.selectbox("Select X-axis", numeric_columns)
    with c2:
        y_axis = st.multiselect("Select Y-axis (Multiple allowed for some plots)", snumeric_columns)
    with c4:
        z_axis = st.selectbox("Select Z-axis (only for 3D Scatter)", numeric_columns) if graph_type == "3D Scatter Plot" else None
    with c3:
        marker = st.selectbox("Select Marker/Line Attribute", numeric_columns)

    if len(numeric_columns) > 1:
        if graph_type == "Correlation Heatmap":
           # st.code("Correlation Heatmap")
            correlation_matrix = data[numeric_columns].corr()
            fig, ax = plt.subplots(figsize=(20, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        elif graph_type == "Distribution Plot":
          #  st.code("Distribution Plot")
            for col in y_axis:
                fig_dist, ax_dist = plt.subplots(figsize=(20, 6))
                sns.histplot(data[col], kde=True, ax=ax_dist)
                ax_dist.set_title(f'Distribution of {col}')
                st.pyplot(fig_dist)

        elif graph_type == "Pairplot":
            #st.code("Pairplot")
            pairplot_fig = sns.pairplot(data[numeric_columns], corner=True)
            st.pyplot(pairplot_fig.fig)

        elif graph_type == "Boxplot":
            #st.code("Boxplot")
            fig_box, ax_box = plt.subplots(figsize=(20, 6))
            sns.boxplot(data=data[y_axis], orient="h", ax=ax_box)
            ax_box.set_title(f'Boxplot for Selected Columns')
            st.pyplot(fig_box)

        elif graph_type == "Violin Plot":
           # st.code("Violin Plot")
            fig_violin, ax_violin = plt.subplots(figsize=(20, 6))
            sns.violinplot(data=data[y_axis], ax=ax_violin)
            ax_violin.set_title('Violin Plot for Selected Columns')
            st.pyplot(fig_violin)

        elif graph_type == "2D Scatter Plot":
          #  st.code("2D Scatter Plot")
            fig_scatter = px.scatter(data, x=x_axis, y=y_axis, color=marker, 
                                     title=f"Scatter Plot of {y_axis} vs {x_axis}")
            st.plotly_chart(fig_scatter)

        elif graph_type == "3D Scatter Plot":
            if len(numeric_columns) >= 3:
                
                fig_scatter_3d = px.scatter_3d(data, x=x_axis, y=y_axis[0], z=z_axis, 
                                               color=marker,
                                               title=f"3D Scatter Plot of {z_axis}, {y_axis[0]}, and {x_axis}")
                st.plotly_chart(fig_scatter_3d,theme='streamlit',selection_mode='point',use_container_width=True)
            else:
                st.warning("Please select at least three columns for the 3D scatter plot.")

        elif graph_type == "Line Plot":
           # st.code("Line Plot")
            fig_line = go.Figure()
            for col in y_axis:
                fig_line.add_trace(go.Scatter(x=data[x_axis], y=data[col], mode='lines+markers', name=col))
            fig_line.update_layout(title="Line Plot for Selected Columns", xaxis_title=x_axis, yaxis_title="Values")
            st.plotly_chart(fig_line)

        elif graph_type == "Density Heatmap":
            
            fig_heatmap = px.density_heatmap(data, x=x_axis, y=y_axis[0], marginal_x="histogram", 
                                             marginal_y="histogram",
                                             title=f"Density Heatmap of {x_axis} and {y_axis[0]}")
            st.plotly_chart(fig_heatmap)

        elif graph_type == "Bubble Chart":
            
            fig_bubble = px.scatter(data, x=x_axis, y=y_axis[0], color=marker,
                                    title=f"Bubble Chart of {y_axis[0]} vs {x_axis} (Marker size based on {marker})")
            st.plotly_chart(fig_bubble)

        elif graph_type == "Hexbin Plot":
            
            fig_hexbin, ax_hexbin = plt.subplots(figsize=(20, 6))
            hb = ax_hexbin.hexbin(data[x_axis], data[y_axis[0]], gridsize=30, cmap='YlOrRd')
            plt.colorbar(hb, ax=ax_hexbin)
            ax_hexbin.set_title(f'Hexbin Plot of {y_axis[0]} vs {x_axis}')
            st.pyplot(fig_hexbin)

        elif graph_type == "Scatter Matrix":
            
            scatter_matrix_fig = px.scatter_matrix(data[numeric_columns])
            scatter_matrix_fig.update_layout(title="Scatter Matrix of Numeric Features", width=2000, height=600)
            st.plotly_chart(scatter_matrix_fig)

    else:
        st.warning("Please select at least two columns for visualization.")
with a:
    
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    

    if uploaded_file:
        
            data = pd.read_csv(uploaded_file)
            st.session_state['data'] = data
            st.session_state['data_loaded'] = True
            st.divider()
            st.subheader("Data Preview") 
            st.write(data.head(10))
            st.divider()
            st.subheader("Data Summary")
            st.write(data.describe())
            st.divider()
            st.subheader("Data visualization")
            plot_advanced_visualizations(data)
            


        
# Ensure data is loaded before proceeding
if st.session_state['data_loaded']:
    data = st.session_state['data']
    
    # Feature and Target selection
    st.subheader("Train and Test classification models")
    st.caption("We will test the following models for classification:")
    i=1
    for model_name,model in models.items():
        
        st.caption(f"({i}) {model_name}")
        i=i+1
    target = st.selectbox('Select Target Variable', data.columns)
    features = st.multiselect('Select Features', data.drop(target, axis=1).columns)
    
    tande=st.button("Train and Evaluate Models")
    if features and target and tande:
        X = data[features]
        y = data[target]
        st.session_state['X'] = X
        st.session_state['y'] = y
        st.session_state['features_selected'] = True
        with st.container(border=True):
            st.success("Features and target variable selected.")
           

# Ensure features and target are set before proceeding
if st.session_state['features_selected']:
    X = st.session_state['X']
    y = st.session_state['y']
    
    if tande:
        st.toast("🛠️ Training and Evaluating Models")
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train and evaluate each model
        accuracy_scores = {}
        errors = {}
        confusion_matrices = {}
        
        model_r=st.expander("### Model Training and Evaluation",expanded=True)

        with model_r:
            for model_name, model in models.items():
                # Cross-validation for accuracy score
                
                scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
                
                
                accuracy = scores.mean()
                accuracy_scores[model_name] = accuracy


                cc=st.container(border=True)
                with cc:
                    st.subheader(f"**{model_name}**")
                
                
                # Train and test the model
                start_time = time.time()
                model.fit(X_train, y_train)
                end_time = time.time()
                training_duration = end_time - start_time
                y_pred = model.predict(X_test)
                error_rate = 1 - accuracy_score(y_test, y_pred)
                errors[model_name] = error_rate
                confusion_matrices[model_name] = confusion_matrix(y_test, y_pred)
                last_time = time.time()
                total_time =last_time-start_time

                
                # Display accuracy and error for each model
                
                
                
                
                with cc:
                    
                    st.success(f"Accuracy: {accuracy:.2f}")
                    st.divider()
                    a,b=st.columns([1,3])
                    with a:
                        
                        st.warning(f"Error Rate: {error_rate:.2f}")
                        st.write(f"Training Duration: {training_duration:.2f} seconds")
                        st.write(f"Total Time: {total_time:.2f} seconds")
                    
                        
                        
                    with b:
                        
                        plt.figure(figsize=(20, 5))
                        sns.scatterplot(x=y_test, y=y_pred, color='blue')
                        plt.xlabel('Real Values')
                        plt.ylabel('Predicted Values')
                        plt.title(f'Predictions versus Real Values')
                        st.pyplot(plt)
                        with st.popover("Confusion Matrix",use_container_width=True):
                            st.write(confusion_matrices[model_name])
            
                
                
        
        # Store results in session state
        
        st.session_state['accuracy_scores'] = accuracy_scores
        st.session_state['errors'] = errors
        st.session_state['model_trained'] = True
        st.toast("✅ Model training and evaluation completed.")

# Plot results only after models are trained
if st.session_state['model_trained']:

    accuracy_scores = st.session_state['accuracy_scores']
    errors = st.session_state['errors']
   
    

    # Select best model based on accuracy
    
    best_model_name = max(accuracy_scores, key=accuracy_scores.get)
    best_model = models[best_model_name]
    
   
    
    # Train the best model on the entire dataset and make predictions
    
    best_model.fit(X, y)
    with st.container(border=True):
        st.subheader(f"Predict using the best model ({best_model_name})",)
        st.info(f"Best Model> {best_model_name} with Accuracy > {accuracy_scores[best_model_name]:.2f}")
    
        # Plot accuracy scores
        
        
        plt.figure(figsize=(20, 5))
        plt.bar(accuracy_scores.keys(), accuracy_scores.values(), color='skyblue')
        plt.ylabel('Accuracy')
        plt.title('Model Accuracy Comparison')
        st.pyplot(plt)
        
        # Plot error rates
        
        plt.figure(figsize=(20, 5))
        plt.bar(errors.keys(), errors.values(), color='salmon')
        plt.ylabel('Error Rate')
        plt.title('Model Error Rate Comparison')
        st.pyplot(plt)
          
        with st.form("Predictions with the Best Model"):  
            input_data = []
            for feature in features:  
                input_val = st.number_input(f"Enter value for {feature}", key=feature,)  
                input_data.append(input_val)  
            submit_button = st.form_submit_button("Predict",type="primary") 

            if submit_button:
                    prediction = predict_with_best_model(input_data)
                    if prediction:
                    
                
                        st.code(f"Predicted {target} : {prediction} ")
                        st.write(f"MODEL USED FOR PREDICTION >  {best_model_name}")
                        st.write(f"ACCURACY SCORE > {accuracy_scores[best_model_name]}")
                    
                    else:
                        st.error("Error in prediction")
                    
