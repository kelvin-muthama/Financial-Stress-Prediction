import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


st.set_page_config(layout='wide',
                   page_title='User Financial Stress Analytical App'
                   )

st.title('User Financial Stress Analytical App')

uploaded_file = st.file_uploader('Upload file here')

st.header('Data Overview')
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df.head(15))

    numeric_cols_train_df = df.select_dtypes(include='number').iloc[:,:-1]

    st.header('Dataframe Summaries')
    if st.button("Dataframe Summaries"):
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Data Types')
            st.write(df.dtypes)
        with col2:
            st.subheader('Statistical Summary')
            st.write(df.describe())

    group_metrics = pd.DataFrame({
        'paybill':      ['volume', 'total_value', 'highest_amount', 'companies'],
        'merchantpay':  ['volume', 'total_value', 'highest_amount', 'merchants'],
        'transfer':     ['volume', 'total_value', 'highest_amount', 'banks'],      
        'mm_send':      ['volume', 'total_value', 'highest_amount', 'recipients'],
        'received':     ['volume', 'total_value', 'highest_amount', 'senders'],
        'deposit':      ['volume', 'total_value', 'highest_amount', 'agents'],
        'withdraw':     ['volume', 'total_value', 'highest_amount', 'agents'],
        'daily_avg':    ['bal', np.nan, np.nan, np.nan],   
    })

    dfs_by_group = {}
    metrics_by_group = {}  # Changed to store dictionaries inside a dictionary

    for group in group_metrics.columns:
        # 1. Filter out main group columns
        g = numeric_cols_train_df.filter(like=group)
        group_df = df[g.columns].copy()
        dfs_by_group[group] = group_df
        
        # Initialize a sub-dictionary for this group's specific metrics
        metrics_by_group[group] = {}
        
        # 2. Extract sub-metrics for this group
        for metric in group_metrics[group]:
            # Skip NaN values (crucial for daily_avg)
            if pd.isna(metric):
                continue
                
            # Filter directly from the already-isolated group dataframe
            metric_cols = group_df.filter(like=metric)
            metrics_by_group[group][metric] = metric_cols.copy()



    totals_df = pd.DataFrame(index=df.index)

    for group_name, sub_metrics in metrics_by_group.items():
        for metric_name, metric_df in sub_metrics.items():
            if len(metric_df.columns) > 0:
                total_column_name = f"{group_name}_{metric_name}_total"
                
                # Calculate the sum and add it to our comparison DataFrame
                totals_df[total_column_name] = metric_df.sum(axis=1)



    summarized_df = df.select_dtypes(include='object').join([totals_df, df[['arpu', 'age', 'x_90_d_activity_rate']]])

    st.header('Summarized DataFrame')
    st.write(summarized_df)

    st.download_button(
    label='Download Summarized DF',
    data=summarized_df.to_csv(index=False),
    file_name='summarized_transactions.csv',
    mime='text/csv'
    )


    st.header('Data Visuals')

    # 1. Use Tabs instead of Buttons so the layout stays open during clicks
    tab1, tab2 = st.tabs(['📊 Numerical Data', '🗂️ Categorical Data'])

    with tab1:
        numeric_cols = summarized_df.select_dtypes(include='number')
        visuals = ['boxplot', 'barplot', 'scatterplot']

        # 2. Control Layout using distinct column structures
        ctrl_col1, ctrl_col2 = st.columns(2)
        with ctrl_col1:
            selected_visual = st.selectbox('Pick a visual style', visuals) 

        if selected_visual == 'boxplot':
            with ctrl_col2:
                selected_x_numeric_col = st.selectbox('Pick a column to visualize', numeric_cols.columns)
            
            # Explicit figure object setup
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.boxplot(data=numeric_cols, x=selected_x_numeric_col, ax=ax) # Fixed to boxplot
            ax.set_title(f'{selected_x_numeric_col} Distribution Boxplot', fontsize=10)
            
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
            
        else:
            # For plots needing two metrics (barplot and scatterplot)
            with ctrl_col2:
                # Create sub-columns within the right side to balance inputs neatly
                sub_col1, sub_col2 = st.columns(2)
                with sub_col1:
                    selected_x_numeric_col = st.selectbox('X-axis column', numeric_cols.columns)
                with sub_col2:
                    selected_y_numeric_col = st.selectbox('Y-axis column', numeric_cols.columns)
            
            fig, ax = plt.subplots(figsize=(5, 3))
            
            if selected_visual == 'barplot':
                sns.barplot(data=numeric_cols, x=selected_x_numeric_col, y=selected_y_numeric_col, ax=ax)
                ax.set_title(f'{selected_x_numeric_col} vs {selected_y_numeric_col} Barplot', fontsize=10)
            elif selected_visual == 'scatterplot':
                sns.scatterplot(data=numeric_cols, x=selected_x_numeric_col, y=selected_y_numeric_col, ax=ax)
                ax.set_title(f'{selected_x_numeric_col} vs {selected_y_numeric_col} Scatter', fontsize=10)
                
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

    with tab2:
        st.write("Place your categorical visualization logic here.")

        cat_cols = df.select_dtypes(include='object').iloc[:,:5]
        visuals = ['histplot', 'piechart', 'barplot']

        col1, col2 = st.columns(2)
        with col1:
            selected_cat_col = st.selectbox('Pick A column to visualize', cat_cols.columns)
        with col2:
            selected_visual = st.selectbox('Pick A column to visualize', visuals)

        # 1. Cleanly check which visualization style the user wants
        if selected_visual == 'histplot':
            # Initialize explicit fig and ax for Matplotlib/Seaborn
            fig = plt.figure(figsize=(6,3))
            
            sns.histplot(data=cat_cols, x=selected_cat_col)
            plt.title(f'{selected_cat_col} Distribution', fontsize=10)
            
            # Render using the specific figure object
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)

        elif selected_visual == 'piechart':
            # Value counts are calculated to give the pie chart numerical slices
            df_counts = cat_cols[selected_cat_col].value_counts().reset_index()
            df_counts.columns = [selected_cat_col, 'count']
            
            # Construct an interactive Plotly Pie chart
            fig = px.pie(
                data_frame=df_counts, 
                names=selected_cat_col, 
                values='count',
                title=f'{selected_cat_col.upper()} Distribution Share'
            )
            
            # Render using Streamlit's native Plotly handler
            st.plotly_chart(fig, use_container_width=True)
