import base64
import streamlit as st
import pandas as pd
import seaborn as sns
import os
import warnings
import string
import re
import nltk
from nltk.stem import WordNetLemmatizer
import plotly.express as px
import requests
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Token and GitHub details
token = "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"
repo_owner = "romero220"
repo_name = "projectmanagement"
branch = "main"

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
warnings.filterwarnings("ignore", message="Converting to PeriodArray/Index representation will drop timezone information")

# Streamlit config
st.set_page_config(page_title="Task Dashboard", layout="wide")

def get_color_palette(num_colors):
    return sns.color_palette("tab10", n_colors=num_colors).as_hex()

@st.cache_data
def load_data():
    csv_files = [file for file in os.listdir('.') if file.endswith('.csv')]
    if not csv_files:
        print("No CSV files found.")
        return pd.DataFrame()

    dataframes = []
    for filename in csv_files:
        df = pd.read_csv(filename)
        numeric_id = filename.split('-')[2] if '-' in filename else 'Unknown'
        df['ProjectID'] = numeric_id
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['ProjectID-ID'] = combined_df['ProjectID'].astype(str) + "-" + combined_df['id'].astype(str)
    combined_df['Full_Name'] = combined_df['user_first_name'].astype(str) + " " + combined_df['user_last_name'].astype(str)
    combined_df['Hours'] = combined_df['minutes'] / 60

    combined_df['task_wo_punct'] = combined_df['task'].apply(lambda x: ''.join([char for char in str(x) if char not in string.punctuation]))
    combined_df['task_wo_punct_split'] = combined_df['task_wo_punct'].apply(lambda x: re.split(r'\W+', str(x).lower()))
    stopword = nltk.corpus.stopwords.words('english')
    combined_df['task_wo_punct_split_wo_stopwords'] = combined_df['task_wo_punct_split'].apply(lambda x: [word for word in x if word not in stopword])
    lemmatizer = WordNetLemmatizer()
    combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] = combined_df['task_wo_punct_split_wo_stopwords'].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])

    return combined_df

# Load data
combined_df = load_data()

# Compute keyword_counts after loading data so it's available globally
if not combined_df.empty:
    keyword_counts = pd.Series(
        {kw: combined_df['task_wo_punct_split_wo_stopwords_lemmatized'].apply(lambda x: kw in x).sum()
         for kw in set([item for sublist in combined_df['task_wo_punct_split_wo_stopwords_lemmatized'] for item in sublist])}
    )
    keyword_counts = keyword_counts.sort_values(ascending=False).reset_index()
    keyword_counts.columns = ['keyword', 'count']
    keyword_counts = keyword_counts[~keyword_counts['keyword'].str.fullmatch(r'\d+')]
    keyword_counts['keyword+count'] = keyword_counts['keyword'] + " (" + keyword_counts['count'].astype(str) + ")"
else:
    keyword_counts = pd.DataFrame(columns=['keyword', 'count', 'keyword+count'])

# Sidebar filters
st.sidebar.header("Filters")
project_ids = combined_df['ProjectID'].dropna().unique()
full_names = combined_df['Full_Name'].dropna().unique()
keyword_options = keyword_counts['keyword+count'].tolist()

ProjectID = st.sidebar.multiselect("Select Project ID", options=project_ids)
keyword_finder = st.sidebar.multiselect("Select a Keyword", options=keyword_options)
date_filter = st.sidebar.date_input("Filter by Date", [])
search_term = st.sidebar.text_input("Search Task", "")
full_name_filter = st.sidebar.multiselect("Filter by Full Name", options=full_names)
time_group = st.sidebar.selectbox("Group by Time Period", options=["Yearly", "Monthly", "Weekly", "Daily"])

# Prepare keyword lookup for filtering
keyword_lookup = {f"{row['keyword']} ({row['count']})": row['keyword'] for _, row in keyword_counts.iterrows()}

# Filter data efficiently
filtered_data = combined_df

if ProjectID:
    filtered_data = filtered_data[filtered_data['ProjectID'].isin(ProjectID)]
if keyword_finder:
    selected_keywords = [keyword_lookup[k] for k in keyword_finder]
    filtered_data = filtered_data[
        filtered_data['task_wo_punct_split_wo_stopwords_lemmatized'].apply(
            lambda x: any(word in x for word in selected_keywords)
        )
    ]
if len(date_filter) == 2:
    filtered_data = filtered_data.copy()
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce").dt.tz_localize(None)
    start_date = pd.to_datetime(date_filter[0])
    end_date = pd.to_datetime(date_filter[1])
    filtered_data = filtered_data[
        (filtered_data["started_at"] >= start_date) & (filtered_data["started_at"] <= end_date)
    ]
if search_term:
    filtered_data = filtered_data[filtered_data['task'].str.contains(search_term, case=False, na=False)]
if full_name_filter:
    filtered_data = filtered_data[filtered_data['Full_Name'].isin(full_name_filter)]

filtered_data = filtered_data.reset_index(drop=True)

# Download filtered data button
csv_data = filtered_data.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label="📥 Download Filtered CSV", data=csv_data, file_name="filtered_data.csv", mime="text/csv")

# File upload and GitHub push
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file is not None:
    file_content = uploaded_file.read()
    file_name = uploaded_file.name
    encoded_content = base64.b64encode(file_content).decode()
    url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents/{file_name}"
    headers = {"Authorization": f"token {token}", "Content-Type": "application/json"}
    data = {"message": f"Adding {file_name} via Streamlit", "content": encoded_content, "branch": branch}

    if st.sidebar.button("Confirm Upload"):
        response = requests.put(url, headers=headers, json=data)
        if response.status_code == 201:
            st.sidebar.success(f"File '{file_name}' uploaded!")
        else:
            st.sidebar.error(f"Upload failed: {response.json().get('message', 'Unknown error')}")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7  = st.tabs([
    "Overview", "📊 Summary", "📈 Visualizations", "⏱️ Task Duration Distribution",
    "👤 User Drilldown", "🏆 Top and Bottom Users ", "☁️ Word Cloud"
])

with tab1:
    st.header("Overview of Data Files")

    def get_file_details():
        files = []
        for file in [f for f in os.listdir('.') if f.endswith('.csv')]:
            try:
                df = pd.read_csv(file)
                files.append({'Filename': file, 'Rows (Excluding Headers)': len(df)})
            except Exception as e:
                st.warning(f"Error reading {file}: {e}")
        return files

    details = get_file_details()
    st.subheader("Uploaded CSV Files")
    st.table(pd.DataFrame(details if details else [{"Filename": "", "Rows (Excluding Headers)": 0}]))

    st.subheader("Preview of Filtered Data (First 100 Rows)")
    st.dataframe(filtered_data.head(100), use_container_width=True)

    st.subheader("Missing Values by Column")
    missing_counts = filtered_data.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if not missing_counts.empty:
        missing_df = pd.DataFrame({'Column': missing_counts.index, 'Missing Values': missing_counts.values})
        fig_missing = px.bar(
            missing_df,
            x='Column',
            y='Missing Values',
            color='Missing Values',
            color_continuous_scale='Reds',
            title="Number of Missing Values per Column",
            labels={'Missing Values': 'Count of NaNs'},
            hover_data={'Column': True}
        )
        fig_missing.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_missing, use_container_width=True)
    else:
        st.success("No missing values in the filtered dataset.")

    # Prepare for time grouping bar chart
    st.subheader(f"Task Count by User - {time_group} View")

    # Convert started_at to datetime once more if needed
    filtered_data["started_at"] = pd.to_datetime(filtered_data["started_at"], errors="coerce")

    # Time grouping based on selection
    if time_group == "Yearly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.year.astype(str)
    elif time_group == "Monthly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%Y-%m')
    elif time_group == "Weekly":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.to_period("W").astype(str)
    elif time_group == "Daily":
        filtered_data["TimeGroup"] = filtered_data["started_at"].dt.strftime('%Y-%m-%d')

    # Group data
    grouped = filtered_data.groupby(["Full_Name", "TimeGroup"])['Hours'].sum().reset_index()

    # Sort time groups chronologically (handles weekly format)
    grouped["TimeGroupSort"] = pd.to_datetime(grouped["TimeGroup"].str.split("/").str[0], errors='coerce')
    grouped = grouped.sort_values("TimeGroupSort")

    # Prepare color palette for the number of unique users in filtered data
    unique_users = grouped['Full_Name'].nunique()
    color_palette = get_color_palette(unique_users)

    # Plotly bar chart
    fig_timegroup = px.bar(
        grouped,
        x="TimeGroup",
        y="Hours",
        color="Full_Name",
        barmode="group",
        title=f"Accumulated Hours per User by {time_group}",
        labels={"TimeGroup": "Time", "Hours": "Total Hours"},
        color_discrete_sequence=color_palette,
        height=500
    )

    # Customize x-axis ticks
    if time_group == "Yearly":
        fig_timegroup.update_xaxes(type="category", tickmode='linear', tickformat='%Y')
    else:
        fig_timegroup.update_xaxes(type="category", tickangle=-45)

    # Display plot
    st.plotly_chart(fig_timegroup, use_container_width=True)

    # Average hours per day of week
    Avarege_hours_day_of_the_week = filtered_data.groupby(filtered_data['started_at'].dt.day_name())['Hours'].mean().reset_index()
    Avarege_hours_day_of_the_week = Avarege_hours_day_of_the_week.rename(columns={'started_at': 'Day'})
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    Avarege_hours_day_of_the_week['Day'] = pd.Categorical(Avarege_hours_day_of_the_week['Day'], categories=days_order, ordered=True)
    Avarege_hours_day_of_the_week = Avarege_hours_day_of_the_week.sort_values('Day')
    fig_avg_hours = px.bar(
        Avarege_hours_day_of_the_week,
        x='Day',
        y='Hours',
        title="Average Hours per Day of the Week",
        labels={'Day': 'Day of the Week', 'Hours': 'Average Hours'},
        color='Hours',
        color_discrete_sequence=px.colors.sequential.Viridis,
        height=500
    )
    st.plotly_chart(fig_avg_hours, use_container_width=True)

with tab2:
    st.subheader("User Summary")
    if not filtered_data.empty:
        user_summary = (
            filtered_data
            .groupby(['user_first_name', 'user_last_name'])
            .agg(
                Total_Minutes=('minutes', 'sum'),
                Task_Count=('minutes', 'count'),
                Avg_Minutes_Per_Task=('minutes', 'mean')
            )
            .reset_index()
            .sort_values(by='Total_Minutes', ascending=False)
        )
        user_summary['Avg_Minutes_Per_Task'] = user_summary['Avg_Minutes_Per_Task'].round(2)
        user_summary.columns = ['First Name', 'Last Name', 'Total Minutes', 'Task Count', 'Avg Minutes/Task']
        st.dataframe(user_summary, use_container_width=True)
        st.download_button(
            label="📥 Download User Summary",
            data=user_summary.to_csv(index=False),
            file_name="user_summary.csv"
        )
        total_minutes = filtered_data['minutes'].sum()
        avg_minutes = filtered_data['minutes'].mean()
        total_tasks = filtered_data.shape[0]
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Time Spent (min)", total_minutes)
        col2.metric("Average Time per Task (min)", round(avg_minutes, 2))
        col3.metric("Total Tasks", total_tasks)
        st.markdown("### 🧠 Insight")
        top_user = user_summary.iloc[0]['First Name'] if not user_summary.empty else None
        top_minutes = user_summary.iloc[0]['Total Minutes'] if not user_summary.empty else None
        top_tasks = user_summary.iloc[0]['Task Count'] if not user_summary.empty else None
        summary_text = (
            f"Across the selected date range, a total of **{total_minutes} minutes** were logged across **{total_tasks} tasks**.\n\n"
            f"Notably, **{top_user}** emerged as the top contributor with **{top_minutes} minutes** spent on **{top_tasks} tasks**, "
            f"suggesting a consistently high engagement level.\n\n"
            f"These metrics offer a holistic view of team workload, individual contribution, and time investment per task."
        )
        st.info(summary_text)
    else:
        st.info("No data to show in User Summary.")

with tab3:
    st.markdown("## 📈 Visualizations")
    if filtered_data.empty:
        st.info("No data to display. Please adjust your filters or upload data.")
    else:
        required_cols = ['user_first_name', 'minutes', 'date', 'task']
        missing_cols = [col for col in required_cols if col not in filtered_data.columns]
        if missing_cols:
            st.error(f"Missing columns in data: {missing_cols}")
        else:
            # Total time per user
            time_chart = filtered_data.groupby('user_first_name')['minutes'].sum().reset_index()
            st.markdown("### Total Time Spent per User")
            fig_time = px.bar(time_chart, x='user_first_name', y='minutes', text='minutes', title='Total Minutes per User')
            st.plotly_chart(fig_time, use_container_width=True)

            st.markdown("---")
            # Minutes per date
            date_chart = filtered_data.groupby('date')['minutes'].sum().reset_index()
            st.markdown("### Time Distribution Over Time")
            fig_date = px.line(date_chart, x='date', y='minutes', markers=True, title='Minutes Logged Over Time')
            st.plotly_chart(fig_date, use_container_width=True)

            st.markdown("---")
            # Breakdown by task type
            task_summary = filtered_data.groupby('task')['minutes'].sum().reset_index().sort_values(by='minutes', ascending=False)
            st.markdown("### Total Minutes by Task Type")
            fig_bar = px.bar(task_summary, x='task', y='minutes', title='Total Minutes by Task Type', text_auto=True)
            fig_bar.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("### 🧠  Insight")
            if not task_summary.empty and not time_chart.empty:
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
            else:
                st.info("Not enough data for insights.")

with tab4:
    st.subheader("Task Duration Distribution and Outliers")
    if not filtered_data.empty and 'minutes' in filtered_data.columns:
        # Histogram
        fig_hist = px.histogram(filtered_data, x='minutes', nbins=30, title="Histogram of Task Durations")
        st.plotly_chart(fig_hist, use_container_width=True)
        # Boxplot
        fig_box = px.box(filtered_data, y='minutes', title="Boxplot of Task Durations")
        st.plotly_chart(fig_box, use_container_width=True)

        # Outlier calculation
        Q1 = filtered_data['minutes'].quantile(0.25)
        Q3 = filtered_data['minutes'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = filtered_data[(filtered_data['minutes'] < lower_bound) | (filtered_data['minutes'] > upper_bound)]

        st.markdown(f"**Number of outliers:** {outliers.shape[0]}")
        cols_to_show = ['date', 'user_first_name', 'task', 'minutes']
        available_cols = [col for col in cols_to_show if col in outliers.columns]
        if not outliers.empty and available_cols:
            st.dataframe(outliers[available_cols], use_container_width=True)
        elif not outliers.empty:
            st.dataframe(outliers, use_container_width=True)
        else:
            st.info("No outlier data available for the selected columns.")

        # Insight Section
        st.markdown("### 🧠  Insight")
        shortest = round(filtered_data['minutes'].min(), 2)
        longest = round(filtered_data['minutes'].max(), 2)
        duration_summary = (
            f"Task durations range widely—from **{shortest} to {longest} minutes**—indicating a mix of quick wins and deeper work.\n\n"
            f"A total of **{outliers.shape[0]} tasks** were flagged as outliers, either exceptionally short or unusually long. "
            f"This could signal errors, bottlenecks, or tasks that deserve process review.\n\n"
            f"Histogram and boxplot distributions help assess whether most tasks fall within acceptable time bands."
        )
        st.info(duration_summary)
    else:
        st.info("No data available for plotting task duration distribution.")

with tab5:
    st.subheader("User Drilldown")
    required_cols = ['user_first_name', 'minutes', 'date', 'task']
    missing_cols = [col for col in required_cols if col not in filtered_data.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {missing_cols}")
        st.write("Available columns:", filtered_data.columns.tolist())
        st.stop()

    users = filtered_data['user_first_name'].dropna().unique()
    if len(users) == 0:
        st.info("No users available for drilldown.")
        st.stop()
    selected_user = st.selectbox("Select User", options=users)
    user_df = filtered_data[filtered_data['user_first_name'] == selected_user]

    col1, col2 = st.columns(2)
    col1.metric("Total Minutes", int(user_df['minutes'].sum()))
    avg_minutes = round(user_df['minutes'].mean(), 2) if not user_df.empty else 0
    col2.metric("Average Task Time", avg_minutes)

    user_chart = user_df.groupby('task', as_index=False)['minutes'].sum()
    fig_user = px.bar(user_chart, x='task', y='minutes', title=f"Task Breakdown for {selected_user}")
    st.plotly_chart(fig_user, use_container_width=True)

    cols_to_show_user = ['date', 'task', 'minutes']
    available_cols_user = [col for col in cols_to_show_user if col in user_df.columns]
    st.dataframe(user_df[available_cols_user], use_container_width=True)

    st.markdown("### 🧠 AI Insight")
    user_task_count = user_df.shape[0]
    user_total = int(user_df['minutes'].sum())

    user_summary = (
        f"**{selected_user}** has completed **{user_task_count} tasks** totaling **{user_total} minutes**. "
        f"Their task time distribution offers insight into workload balance.\n\n"
        f"If one task type dominates, it may highlight specialization or possible over-dependence on this user for certain duties.\n\n"
        f"Use this view to evaluate both individual performance and role focus."
    )
    st.info(user_summary)

with tab6:
    st.header("🏆 Top & Bottom 5 User Time Stats")

    required_cols = ['user_first_name', 'minutes']
    missing_cols = [col for col in required_cols if col not in filtered_data.columns]
    if missing_cols:
        st.error(f"Missing columns in data: {missing_cols}")
        st.write("Available columns:", filtered_data.columns.tolist())
        st.stop()

    # Total time per user
    total_time = filtered_data.groupby('user_first_name')['minutes'].sum().reset_index()
    avg_time = filtered_data.groupby('user_first_name')['minutes'].mean().reset_index()

    # Top and bottom 5 by total time spent
    top5_total = total_time.sort_values('minutes', ascending=False).head(5)
    bottom5_total = total_time.sort_values('minutes', ascending=True).head(5)

    # Top and bottom 5 by average time spent
    top5_avg = avg_time.sort_values('minutes', ascending=False).head(5)
    bottom5_avg = avg_time.sort_values('minutes', ascending=True).head(5)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 5 Users by Total Time Spent")
        st.dataframe(top5_total.rename(columns={'user_first_name': 'User', 'minutes': 'Total Minutes'}), use_container_width=True)
        fig1 = px.bar(top5_total, x='user_first_name', y='minutes',
                      title='Top 5 Users by Total Time Spent', labels={'user_first_name': 'User', 'minutes': 'Total Minutes'})
        st.plotly_chart(fig1, use_container_width=True)

        st.subheader("Top 5 Users by Average Time Spent")
        st.dataframe(top5_avg.rename(columns={'user_first_name': 'User', 'minutes': 'Avg Minutes'}), use_container_width=True)
        fig3 = px.bar(top5_avg, x='user_first_name', y='minutes',
                      title='Top 5 Users by Average Time Spent', labels={'user_first_name': 'User', 'minutes': 'Avg Minutes'})
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Lowest 5 Users by Total Time Spent")
        st.dataframe(bottom5_total.rename(columns={'user_first_name': 'User', 'minutes': 'Total Minutes'}), use_container_width=True)
        fig2 = px.bar(bottom5_total, x='user_first_name', y='minutes',
                      title='Lowest 5 Users by Total Time Spent', labels={'user_first_name': 'User', 'minutes': 'Total Minutes'})
        st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Lowest 5 Users by Average Time Spent")
        st.dataframe(bottom5_avg.rename(columns={'user_first_name': 'User', 'minutes': 'Avg Minutes'}), use_container_width=True)
        fig4 = px.bar(bottom5_avg, x='user_first_name', y='minutes',
                      title='Lowest 5 Users by Average Time Spent', labels={'user_first_name': 'User', 'minutes': 'Avg Minutes'})
        st.plotly_chart(fig4, use_container_width=True)

with tab7:
    st.subheader("Word Cloud of Tasks")

    # Generate text for word cloud
    text = " ".join(filtered_data['task'].dropna().astype(str).values)
    if not text.strip():
        st.info("No task data available for word cloud.")
    else:
        wc = WordCloud(width=800, height=400, background_color="white").generate(text)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(wc, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # AI Insight
    st.markdown("### 🧠 Insight")
    task_counts = filtered_data['task'].value_counts()
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
