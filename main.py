import streamlit as st
import pandas as pd
import os
from plotly import graph_objs as go
from prophet import Prophet


UPLOAD_DIR = "uploaded_datasets"


os.makedirs(UPLOAD_DIR, exist_ok=True)

st.title("Daily Sales Analysis with Moving Average")


uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"], key="upload_file")
if uploaded_file:
    
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"Uploaded file saved as: {uploaded_file.name}")


files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".csv")]
if files:
    dataset = st.selectbox("Select Dataset for Analysis", files)

    
    data_path = os.path.join(UPLOAD_DIR, dataset)
    data = pd.read_csv(data_path, parse_dates=["Date"])
    data.sort_values(by="Date", inplace=True)

   
    st.subheader("Raw Data")
    st.write(data)

    
    ma_window = st.radio("Select Moving Average Window (Days)", [7, 14, 30])

    
    data["Moving Average"] = data["Total Sales"].rolling(window=ma_window).mean()

    
    st.subheader("Time-Series Data")
    date_range = st.slider(
        "Select Time Range",
        min_value=data["Date"].min().to_pydatetime(),
        max_value=data["Date"].max().to_pydatetime(),
        value=(data["Date"].min().to_pydatetime(), data["Date"].max().to_pydatetime()),
        format="MMM YYYY"
    )

    
    st.markdown(f"**Selected Time Range:** {date_range[0].strftime('%B %d, %Y')} - {date_range[1].strftime('%B %d, %Y')}")

    
    filtered_data = data[(data["Date"] >= date_range[0]) & (data["Date"] <= date_range[1])]

    
    fig = go.Figure()

   
    fig.add_trace(go.Scatter(
        x=filtered_data["Date"],
        y=filtered_data["Total Sales"],
        mode="lines+markers",
        name="Total Sales",
        line=dict(color="blue"),
        hovertemplate="<b>Date:</b> %{x}<br><b>Total Sales:</b> %{y}<extra></extra>"
    ))

    
    fig.add_trace(go.Scatter(
        x=filtered_data["Date"],
        y=filtered_data["Moving Average"],
        mode="lines+markers",
        name=f"{ma_window}-Day Moving Average",
        line=dict(color="orange"),
        hovertemplate="<b>Date:</b> %{x}<br><b>Moving Average:</b> %{y:.2f}<extra></extra>"
    ))

    fig.update_layout(
        title="Time-Series Plot",
        xaxis_title="Date",
        yaxis_title="Sales",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=30, label="1m", step="day", stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )
    st.plotly_chart(fig)

    
    prophet_data = data.rename(columns={"Date": "ds", "Total Sales": "y"})
    model = Prophet(weekly_seasonality=True, yearly_seasonality=True)
    model.fit(prophet_data)

    
    future = model.make_future_dataframe(periods=365, freq="D")
    forecast = model.predict(future)

    
    st.subheader("Forecasted Sales")
    forecast_fig = go.Figure()

   
    forecast_fig.add_trace(go.Scatter(
        x=prophet_data["ds"],
        y=prophet_data["y"],
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color="blue"),
        hovertemplate="<b>Date:</b> %{x}<br><b>Actual Sales:</b> %{y}<extra></extra>"
    ))

    
    forecast_fig.add_trace(go.Scatter(
        x=forecast["ds"],
        y=forecast["yhat"],
        mode="lines",
        name="Forecasted Sales",
        line=dict(color="green"),
        hovertemplate="<b>Date:</b> %{x}<br><b>Forecast:</b> %{y}<extra></extra>"
    ))

    forecast_fig.update_layout(
        title="Sales Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        xaxis=dict(rangeslider=dict(visible=True), type="date")
    )
    st.plotly_chart(forecast_fig)

    
    st.subheader("Forecast Components")
    components_fig = model.plot_components(forecast)
    st.pyplot(components_fig)

else:
    st.warning("No datasets available. Please upload a CSV file.")
