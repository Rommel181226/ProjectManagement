import streamlit as st
import pandas as pd
import plotly.express as px
import calplot
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Task Dashboard", layout="wide")
st.title("🗂️ Task Time Analysis Dashboard")

# Sidebar - Logo and Title
logo_path = os.path.join("images", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
st.sidebar.markdown("## 📁 Task Dashboard Sidebar")

# Upload CSV Files
uploaded_files = st.sidebar.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

@st.cache_data
def load_all_data(files):
    combined = []
    for file in files:
        df = pd.read_csv(file)
        df['started_at'] = pd.to_datetime(df['started_at'], errors='coerce')
        df['date'] = df['started_at'].dt.date
        df['hour'] = df['started_at'].dt.hour
        combined.append(df)
    return pd.concat(combined, ignore_index=True)

# Main logic
if uploaded_files:
    df = load_all_data(uploaded_files)

    # Sidebar filters (only user and date)
    users = df['user_first_name'].dropna().unique()
    min_date, max_date = df['date'].min(), df['date'].max()

    st.sidebar.subheader("Filter Data")
    selected_users = st.sidebar.multiselect("User", options=users, default=list(users))
    selected_dates = st.sidebar.date_input("Date Range", [min_date, max_date])

    # Apply filters
    mask = (
        df['user_first_name'].isin(selected_users) &
        (df['date'] >= selected_dates[0]) & (df['date'] <= selected_dates[1])
    )
    filtered_df = df[mask]

    # Tabs: 7 Total
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Summary", "📈 Visualizations", "📋 Task Records",
        "👤 User Drilldown", "☁️ Word Cloud", "📅 Calendar Heatmap", "🗂️ All Uploaded Data"
    ])

    with tab1:
        st.subheader("User Summary")
        
        total_minutes = filtered_df['minutes'].sum()
        avg_minutes = filtered_df['minutes'].mean()
        total_tasks = filtered_df.shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Time Spent (min)", total_minutes)
        col2.metric("Average Time per Task (min)", round(avg_minutes, 2))
        col3.metric("Total Tasks", total_tasks)
        
        user_summary = (
            filtered_df
            .groupby(['user_first_name', 'user_last_name', 'user_locale'])
            .agg(
                Total_Minutes=('minutes', 'sum'),
                Task_Count=('minutes', 'count'),
                Avg_Minutes_Per_Task=('minutes', 'mean')
            )
            .reset_index()
            .sort_values(by='Total_Minutes', ascending=False)
        )
        user_summary['Avg_Minutes_Per_Task'] = user_summary['Avg_Minutes_Per_Task'].round(2)
        user_summary.columns = ['First Name', 'Last Name', 'Locale', 'Total Minutes', 'Task Count', 'Avg Minutes/Task']
        st.dataframe(user_summary, use_container_width=True)

        st.download_button(
            label="📥 Download User Summary",
            data=user_summary.to_csv(index=False),
            file_name="user_summary.csv"
        )

        
    with tab2:
        st.markdown("### Time Spent per User")
        time_chart = filtered_df.groupby('user_first_name')['minutes'].sum().reset_index()
        fig_time = px.bar(time_chart, x='user_first_name', y='minutes', title='Total Minutes per User')
        st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("### Time Distribution by Date")
        date_chart = filtered_df.groupby('date')['minutes'].sum().reset_index()
        fig_date = px.line(date_chart, x='date', y='minutes', markers=True, title='Minutes Logged Over Time')
        st.plotly_chart(fig_date, use_container_width=True)

        st.subheader("Breakdown by Task Type")
        task_summary = filtered_df.groupby('task')['minutes'].sum().reset_index().sort_values(by='minutes', ascending=False)

        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(task_summary, names='task', values='minutes', title="Total Minutes by Task Type")
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.bar(task_summary, x='task', y='minutes', title='Total Minutes by Task Type', text_auto=True)
            st.plotly_chart(fig_bar, use_container_width=True)

    with tab3:
        st.markdown("### Task Records")
        st.dataframe(filtered_df[['date', 'user_first_name', 'user_last_name', 'task', 'minutes']], use_container_width=True)
        st.download_button("📥 Download Filtered Data", data=filtered_df.to_csv(index=False), file_name="filtered_data.csv")

    with tab4:
        st.subheader("User Drilldown")
        selected_user = st.selectbox("Select User", options=filtered_df['user_first_name'].unique())
        user_df = filtered_df[filtered_df['user_first_name'] == selected_user]

        col1, col2 = st.columns(2)
        col1.metric("Total Minutes", user_df['minutes'].sum())
        col2.metric("Average Task Time", round(user_df['minutes'].mean(), 2))

        user_chart = user_df.groupby('task')['minutes'].sum().reset_index()
        fig_user = px.bar(user_chart, x='task', y='minutes', title=f"Task Breakdown for {selected_user}")
        st.plotly_chart(fig_user, use_container_width=True)

        st.markdown("### Task History")
        st.dataframe(user_df[['date', 'task', 'minutes']], use_container_width=True)

    with tab5:
        st.subheader("☁️ Word Cloud of Task Names")
        task_weights = filtered_df.groupby('task')['minutes'].sum().to_dict()
        wordcloud = WordCloud(width=1000, height=400, background_color='white').generate_from_frequencies(task_weights)

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        st.pyplot(fig)

    with tab6:
        st.subheader("📅 Calendar Heatmap")
        heatmap_data = filtered_df.groupby('date')['minutes'].sum().reset_index()
        heatmap_data['date'] = pd.to_datetime(heatmap_data['date'])
        heatmap_series = heatmap_data.set_index('date')['minutes']

        fig, ax = calplot.calplot(
            heatmap_series,
            cmap='YlGn',
            colorbar=True,
            figsize=(16, 8),
            suptitle='Minutes Logged per Day'
        )
        st.pyplot(fig)

    with tab7:
        st.subheader("🗂️ All Uploaded Data (Unfiltered)")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="📥 Download All Uploaded Data",
            data=df.to_csv(index=False),
            file_name="all_uploaded_data.csv"
        )

else:
    st.info("Upload one or more CSV files to begin.")
