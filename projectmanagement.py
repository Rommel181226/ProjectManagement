import streamlit as st
import pandas as pd
import plotly.express as px
import calplot
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_page_config(page_title="Task Dashboard", layout="wide")
st.title("🗂️ Task Time Analysis Dashboard")

logo_path = os.path.join("images", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)
st.sidebar.markdown("## 📁 Task Dashboard Sidebar")

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

if uploaded_files:
    df = load_all_data(uploaded_files)

    users = df['user_first_name'].dropna().unique()
    min_date, max_date = df['date'].min(), df['date'].max()

    st.sidebar.subheader("Filter Data")
    selected_users = st.sidebar.multiselect("User", options=users, default=list(users))
    selected_dates = st.sidebar.date_input("Date Range", [min_date, max_date])

    mask = (
        df['user_first_name'].isin(selected_users) &
        (df['date'] >= selected_dates[0]) & (df['date'] <= selected_dates[1])
    )
    filtered_df = df[mask]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Summary", "📈 Visualizations", "⏱️ Task Duration Distribution",
        "👤 User Drilldown", "☁️ Word Cloud", "📅 Calendar Heatmap",
        "📑 All Uploaded Data", "👥 User Comparison Dashboard"
    ])

    with tab1:
        st.subheader("User Minutes and Info")
        minutes_table = filtered_df.groupby('user_first_name')['minutes'].sum().reset_index()
        user_info = df[['user_first_name', 'user_last_name', 'user_locale']].drop_duplicates()
        summary_table = pd.merge(minutes_table, user_info, on='user_first_name', how='left')
        summary_table = summary_table.rename(columns={'minutes': 'total_minutes'})
        st.dataframe(summary_table.sort_values(by='total_minutes', ascending=False), use_container_width=True)

        total_minutes = filtered_df['minutes'].sum()
        avg_minutes = filtered_df['minutes'].mean()
        total_tasks = filtered_df.shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Time Spent (min)", total_minutes)
        col2.metric("Average Time per Task (min)", round(avg_minutes, 2))
        col3.metric("Total Tasks", total_tasks)

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
        st.subheader("Task Duration Distribution")
        fig_hist = px.histogram(
            filtered_df,
            x='minutes',
            nbins=30,
            title="Histogram of Task Durations (minutes)",
            labels={"minutes": "Task Duration (minutes)"}
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        fig_box = px.box(
            filtered_df,
            y='minutes',
            title="Boxplot of Task Durations",
            labels={"minutes": "Task Duration (minutes)"}
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.subheader("Outlier Detection (Task Duration)")
        Q1 = filtered_df['minutes'].quantile(0.25)
        Q3 = filtered_df['minutes'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        st.write(f"Outlier thresholds: Tasks shorter than {lower_bound:.2f} minutes or longer than {upper_bound:.2f} minutes.")

        outliers = filtered_df[(filtered_df['minutes'] < lower_bound) | (filtered_df['minutes'] > upper_bound)]

        if outliers.empty:
            st.success("No outliers detected in task durations.")
        else:
            st.warning(f"Detected {outliers.shape[0]} outlier tasks:")
            st.dataframe(outliers[['date', 'user_first_name', 'task', 'minutes']], use_container_width=True)

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
        st.subheader("Word Cloud of Tasks")
        text = " ".join(filtered_df['task'].dropna().astype(str).values)
        if text.strip() == "":
            st.info("No task data available for word cloud.")
        else:
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

    with tab6:
        st.subheader("Calendar Heatmap")
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
        st.subheader("All Uploaded Data (Before Filtering)")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            label="📥 Download All Uploaded Data",
            data=df.to_csv(index=False),
            file_name="compiled_uploaded_data.csv"
        )

    with tab8:
        st.subheader("User Comparison Dashboard")
        user_stats = filtered_df.groupby('user_first_name').agg(
            total_minutes=('minutes', 'sum'),
            task_count=('task', 'count'),
            avg_task_time=('minutes', 'mean')
        ).reset_index()

        user_stats['avg_task_time'] = user_stats['avg_task_time'].round(2)

        st.dataframe(user_stats.sort_values(by='total_minutes', ascending=False), use_container_width=True)

        fig = px.bar(
            user_stats.sort_values(by='total_minutes', ascending=False),
            x='user_first_name',
            y=['total_minutes', 'task_count', 'avg_task_time'],
            barmode='group',
            title="User Performance Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Upload one or more CSV files to begin.")
