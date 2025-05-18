import streamlit as st
import pandas as pd
import plotly.express as px
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(page_title="Task Dashboard", layout="wide")
st.title("📂 Task Time Analysis Dashboard")

# Sidebar logo (optional)
logo_path = os.path.join("images", "logo.png")
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width=150)

st.sidebar.markdown("## 📁 Task Dashboard Sidebar")

# File uploader widget
uploaded_files = st.sidebar.file_uploader(
    "Upload CSV files", type=["csv"], accept_multiple_files=True
)

# Sidebar Enhancements
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Navigation")
st.sidebar.markdown("Use the tabs above to explore summaries, visualizations, comparisons, and insights.")
st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 Tips")
st.sidebar.markdown("- Filter by user and date to focus the analysis.\n- Use the word cloud for keyword trends.\n- Review outliers to detect anomalies.")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📘 About")
st.sidebar.info("This dashboard helps visualize and analyze task duration data. Built with Streamlit, Plotly, and AI-generated insights.")

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

    st.sidebar.subheader("🔧 Filter Data")
    selected_users = st.sidebar.multiselect("User", options=users, default=list(users))
    selected_dates = st.sidebar.date_input("Date Range", [min_date, max_date])

    mask = (
        df['user_first_name'].isin(selected_users) &
        (df['date'] >= selected_dates[0]) & (df['date'] <= selected_dates[1])
    )
    filtered_df = df[mask]

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "📊 Summary", "📈 Visualizations", "⏱️ Task Duration Distribution",
        "👤 User Drilldown", "🏆 Top Users", "☁️ Word Cloud", "📅 Calendar Heatmap",
        "📁 All Uploaded Data", "👥 User Comparison", "🕒 Hourly Heatmap"
    ])

    
    # --- Tab 1 ---
    with tab1:
        st.subheader("User Summary")
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

        total_minutes = filtered_df['minutes'].sum()
        avg_minutes = filtered_df['minutes'].mean()
        total_tasks = filtered_df.shape[0]

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Time Spent (min)", total_minutes)
        col2.metric("Average Time per Task (min)", round(avg_minutes, 2))
        col3.metric("Total Tasks", total_tasks)

        st.markdown("### 🧠 AI Insight")
        top_user = user_summary.iloc[0]['First Name']
        top_minutes = user_summary.iloc[0]['Total Minutes']
        top_tasks = user_summary.iloc[0]['Task Count']
        avg_task_time = user_summary['Avg Minutes/Task'].mean()

        summary_text = (
            f"Across the selected date range, a total of **{total_minutes} minutes** were logged across **{total_tasks} tasks** "
            f"by **{len(selected_users)} users**. The average task duration was **{round(avg_minutes, 2)} minutes**.\n\n"
            f"Notably, **{top_user}** emerged as the top contributor with **{top_minutes} minutes** spent on **{top_tasks} tasks**, "
            f"suggesting a consistently high engagement level.\n\n"
            f"These metrics offer a holistic view of team workload, individual contribution, and time investment per task."
            )
        st.info(summary_text)

    # --- Tab 2 ---
    with tab2:
        st.markdown("## 📈 Visualizations")

        time_chart = filtered_df.groupby('user_first_name')['minutes'].sum().reset_index()
        st.markdown("### Total Time Spent per User")
        fig_time = px.bar(time_chart, x='user_first_name', y='minutes', text='minutes', title='Total Minutes per User')
        fig_time.update_traces(texttemplate='%{text:.2s}', textposition='outside')
        st.plotly_chart(fig_time, use_container_width=True)

        st.markdown("---")
        st.markdown("### Time Distribution Over Time")
        date_chart = filtered_df.groupby('date')['minutes'].sum().reset_index()
        fig_date = px.line(date_chart, x='date', y='minutes', markers=True, title='Minutes Logged Over Time')
        st.plotly_chart(fig_date, use_container_width=True)

        st.markdown("---")
        st.markdown("### Breakdown by Task Type")
        task_summary = filtered_df.groupby('task')['minutes'].sum().reset_index().sort_values(by='minutes', ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            fig_pie = px.pie(task_summary, names='task', values='minutes', title="Total Minutes by Task Type", hole=0.3)
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        with col2:
            fig_bar = px.bar(task_summary, x='task', y='minutes', title='Total Minutes by Task Type', text_auto=True)
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("### 🧠 AI Insight")
        top_task = task_summary.iloc[0]['task']
        top_task_minutes = task_summary.iloc[0]['minutes']
        most_active_user = time_chart.sort_values(by='minutes', ascending=False).iloc[0]['user_first_name']

        viz_summary = (
            f"From the visual analysis, **{top_task}** stands out as the most time-consuming task type, "
            f"accumulating **{int(top_task_minutes)} minutes**. This may indicate complexity or frequent repetition.\n\n"
            f"**{most_active_user}** leads in total time logged, suggesting either a higher workload or a more time-intensive role.\n\n"
            f"The temporal line chart helps identify productivity trends—spikes may reflect sprints or deadlines, while dips could indicate downtime or under-reporting."
            )
        st.info(viz_summary)


    # --- Tab 3 ---
    with tab3:
        st.subheader("Task Duration Distribution")
        fig_hist = px.histogram(filtered_df, x='minutes', nbins=30, title="Histogram of Task Durations")
        st.plotly_chart(fig_hist, use_container_width=True)
        fig_box = px.box(filtered_df, y='minutes', title="Boxplot of Task Durations")
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Outlier Detection")
        Q1 = filtered_df['minutes'].quantile(0.25)
        Q3 = filtered_df['minutes'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        st.write(f"Outlier thresholds: < {lower_bound:.2f} or > {upper_bound:.2f} minutes")

        outliers = filtered_df[(filtered_df['minutes'] < lower_bound) | (filtered_df['minutes'] > upper_bound)]
        if outliers.empty:
            st.success("No outliers detected.")
        else:
            st.warning(f"Found {outliers.shape[0]} outlier tasks")
            st.dataframe(outliers[['date', 'user_first_name', 'task', 'minutes']], use_container_width=True)

        st.markdown("### 🧠 AI Insight")
        shortest = round(filtered_df['minutes'].min(), 2)
        longest = round(filtered_df['minutes'].max(), 2)

        duration_summary = (
            f"Task durations range widely—from **{shortest} to {longest} minutes**—indicating a mix of quick wins and deeper work.\n\n"
            f"A total of **{outliers.shape[0]} tasks** were flagged as outliers, either exceptionally short or unusually long. "
            f"This could signal errors, bottlenecks, or tasks that deserve process review.\n\n"
            f"Histogram and boxplot distributions help assess whether most tasks fall within acceptable time bands."
             )
        st.info(duration_summary)

    # --- Tab 4 ---
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
        st.dataframe(user_df[['date', 'task', 'minutes']], use_container_width=True)

        st.markdown("### 🧠 AI Insight")
        user_task_count = user_df.shape[0]
        user_total = user_df['minutes'].sum()

        user_summary = (
            f"**{selected_user}** has completed **{user_task_count} tasks** totaling **{user_total} minutes**. "
            f"Their task time distribution offers insight into workload balance.\n\n"
            f"If one task type dominates, it may highlight specialization or possible over-dependence on this user for certain duties.\n\n"
            f"Use this view to evaluate both individual performance and role focus."
             )
        st.info(user_summary)

    # --- Tab 5 ---
    with tab5:
        st.subheader("🏆 Top 5 Users by Total Time Spent")
        top_users_df = (
            filtered_df
            .groupby('user_first_name')['minutes']
            .sum()
            .reset_index()
            .sort_values(by='minutes', ascending=False)
            .head(5)
        )

        fig_top_users = px.bar(
            top_users_df,
            x='user_first_name',
            y='minutes',
            text='minutes',
            title='Top 5 Users by Total Minutes Logged',
            color='user_first_name'
        )
        fig_top_users.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_top_users.update_layout(showlegend=False)
        st.plotly_chart(fig_top_users, use_container_width=True)

        # 🧠 Enhanced AI Insight with percentage share
        total_all_users_time = filtered_df['minutes'].sum()
        top_users_df['percentage'] = (top_users_df['minutes'] / total_all_users_time * 100).round(2)

        most_active = top_users_df.iloc[0]
        avg_top5 = round(top_users_df['minutes'].mean(), 2)
        highest_pct = most_active['percentage']

        insight_text = (
            f"Among the top 5 users, **{most_active['user_first_name']}** logged the highest time with "
            f"**{int(most_active['minutes'])} minutes**, accounting for **{highest_pct}%** of total logged time.\n\n"
            f"The average time spent across the top 5 users is approximately **{avg_top5} minutes**.\n\n"
        )

        if highest_pct > 40:
            insight_text += (
                "⚠️ This user is contributing a disproportionately large share of total time. "
                "Consider reviewing task assignments or workload distribution to prevent burnout or dependency."
            )
        else:
            insight_text += (
                "✅ The time distribution among top users appears balanced, suggesting a healthy workload spread."
            )
        st.markdown("### 🧠 AI Insight")
        st.info(insight_text)
    
    with tab6:
        st.subheader("Word Cloud of Tasks")
        text = " ".join(filtered_df['task'].dropna().astype(str).values)
        if not text.strip():
            st.info("No task data available for word cloud.")
        else:
            wc = WordCloud(width=800, height=400, background_color="white").generate(text)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.imshow(wc, interpolation='bilinear')
            ax.axis("off")
            st.pyplot(fig)

        st.markdown("### 🧠 AI Insight")
        task_counts = filtered_df['task'].value_counts()
        top_wc_task = task_counts.index[0] if not task_counts.empty else None

        wc_summary = (
            f"The word cloud visualizes the most frequently logged tasks. "
            f"**{top_wc_task}** appears most often, suggesting it's central to team operations.\n\n"
            f"Frequent mentions may reflect routine responsibilities, while missing or rare task types could indicate under-reporting "
            f"or areas with less activity.\n\n"
            f"Use this to understand recurring themes or evaluate if task tracking is comprehensive."
        if top_wc_task else "No task text was available to analyze frequency trends."
            )
        st.info(wc_summary)


    # --- Tab 7 ---
    with tab7:
        st.subheader("📅 Calendar Heatmap")
        calendar_df = filtered_df.copy()
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        daily_summary = calendar_df.groupby('date')['minutes'].sum().reset_index()

        fig = px.density_heatmap(daily_summary, x='date', y='minutes', nbinsx=60,
                                 title='Minutes Logged per Day', color_continuous_scale='YlOrBr')
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧠 AI Insight")
        busiest_day = daily_summary.sort_values(by='minutes', ascending=False).iloc[0]

        cal_summary = (
            f"The calendar heatmap reveals activity trends across dates. The peak productivity day was **{busiest_day['date'].strftime('%Y-%m-%d')}**, "
            f"logging **{int(busiest_day['minutes'])} minutes**.\n\n"
            f"Clusters of high activity can correspond to deadlines, launches, or team pushes. "
            f"Meanwhile, extended low-activity periods may point to gaps in task tracking, team absences, or lulls in workload.\n\n"
            f"Use this to inform capacity planning or investigate performance cycles."
            )
        st.info(cal_summary)

    # --- Tab 8 ---
    with tab8:
        st.subheader("Raw Uploaded Data")
        st.dataframe(filtered_df, use_container_width=True)
        st.markdown("### 🧠 AI Insight")
        st.info(
            f"The dataset includes **{filtered_df.shape[0]} tasks** across **{filtered_df['user_first_name'].nunique()} users**.\n\n"
            f"This raw view is helpful for audits, exports, or deep data exploration. "
            f"Filtering allows for precise investigation of specific time ranges or user activity."
            )

    # --- Tab 9 ---
    with tab9:
        st.subheader("User Comparison")
        comp_users = st.multiselect("Select Users", options=users, default=users[:2])
        if len(comp_users) < 2:
            st.info("Select at least 2 users.")
        else:
            comp_df = filtered_df[filtered_df['user_first_name'].isin(comp_users)]
            comp_summary = comp_df.groupby(['user_first_name', 'task'])['minutes'].sum().reset_index()
            fig = px.bar(comp_summary, x='task', y='minutes', color='user_first_name', barmode='group')
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("### 🧠 AI Insight")
        if not comp_df.empty:
            top_user = comp_df.groupby('user_first_name')['minutes'].sum().idxmax()
            top_value = comp_df.groupby('user_first_name')['minutes'].sum().max()
            st.info(
            f"Among the compared users, **{top_user}** has logged the most time with **{top_value} minutes**.\n\n"
            f"This comparison helps detect imbalances in task distribution, identify high performers, or understand who is handling which task types."
            )


    # --- Tab 10 ---
    with tab10:
    st.subheader("📈 Hourly Activity Line Graph")
    if 'hour' in filtered_df.columns:
        line_df = filtered_df.copy()
        line_df['day'] = pd.to_datetime(line_df['date']).dt.day_name()
        line_df['hour'] = line_df['hour'].astype(int)

        # Group and sum minutes by hour and day
        grouped = line_df.groupby(['day', 'hour'])['minutes'].sum().reset_index()

        # Order days manually
        ordered_days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        grouped['day'] = pd.Categorical(grouped['day'], categories=ordered_days, ordered=True)
        grouped = grouped.sort_values(['day', 'hour'])

        # Plot line chart
        fig = px.line(
            grouped,
            x='hour',
            y='minutes',
            color='day',
            markers=True,
            labels={'hour': 'Hour of Day', 'minutes': 'Total Minutes'},
            title='Hourly Activity Trends by Day'
        )
        fig.update_layout(xaxis=dict(dtick=1), height=500)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧠 AI Insight")
        peak_hour = grouped.groupby('hour')['minutes'].sum().idxmax()
        st.info(
            f"The line chart shows that the peak hour across the week is **{peak_hour}:00**. "
            f"This suggests consistent team engagement during this hour.\n\n"
            f"Observe dips or plateaus for potential process improvements or break timings."
        )

    else:
        st.info("Please upload one or more CSV files to get started.")
