import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stMetric {
        background-color: white !important;
        padding: 1rem !important;
        border-radius: 0.5rem !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    .stMetric label {
        color: #1f1f1f !important;
        font-weight: 600 !important;
    }
    .stMetric [data-testid="stMetricValue"] {
        color: #1f77b4 !important;
        font-size: 1.8rem !important;
        font-weight: bold !important;
    }
    .stMetric [data-testid="stMetricDelta"] {
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data(ttl=3600)
def fetch_stores():
    """Fetch available stores from API"""
    try:
        response = requests.get(f"{API_URL}/stores")
        if response.status_code == 200:
            return response.json()["stores"]
        return []
    except Exception as e:
        st.error(f"Error fetching stores: {str(e)}")
        return []

@st.cache_data(ttl=3600)
def fetch_items(store_id):
    """Fetch items for a specific store"""
    try:
        response = requests.get(f"{API_URL}/items/{store_id}")
        if response.status_code == 200:
            return response.json()["items"]
        return []
    except Exception as e:
        st.error(f"Error fetching items: {str(e)}")
        return []

def fetch_forecast(store_id, item_id):
    """Fetch forecast data"""
    try:
        response = requests.post(
            f"{API_URL}/forecast",
            json={"store_id": store_id, "item_id": item_id}
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Error fetching forecast: {str(e)}")
        return None

def fetch_historical(store_id, item_id, days=90):
    """Fetch historical sales data"""
    try:
        response = requests.get(
            f"{API_URL}/historical/{store_id}/{item_id}?days={days}"
        )
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Error fetching historical data: {str(e)}")
        return None

def create_forecast_plot(historical_data, forecast_data):
    """Create interactive forecast visualization with 3 months historical + 30 days forecast"""
    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=["📊 Sales History (Last 3 Months) & 30-Day Forecast"]
    )
    
    # Convert dates to datetime objects for proper plotting
    hist_dates = None
    forecast_dates = None
    
    if historical_data:
        hist_dates = pd.to_datetime(historical_data['dates'])
    
    if forecast_data:
        forecast_dates = pd.to_datetime(forecast_data['dates'])
    
    # Add historical data
    if historical_data and hist_dates is not None:
        fig.add_trace(
            go.Scatter(
                x=hist_dates,
                y=historical_data['sales'],
                mode='lines+markers',
                name='Historical Sales',
                line=dict(color='#3498db', width=3),
                marker=dict(size=5, symbol='circle'),
                hovertemplate='<b>Historical</b><br>Date: %{x}<br>Sales: %{y:.2f} units<extra></extra>',
                fill='tonexty',
                fillcolor='rgba(52, 152, 219, 0.1)'
            )
        )
    
    # Add forecast data
    if forecast_data and forecast_dates is not None:
        fig.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast_data['forecasted_sales'],
                mode='lines+markers',
                name='30-Day Forecast',
                line=dict(color='#e74c3c', width=3, dash='dash'),
                marker=dict(size=6, symbol='diamond'),
                hovertemplate='<b>Forecast</b><br>Date: %{x}<br>Predicted: %{y:.2f} units<extra></extra>',
                fill='tonexty',
                fillcolor='rgba(231, 76, 60, 0.1)'
            )
        )
        
        # Add confidence band (simple estimate)
        if len(forecast_data['forecasted_sales']) > 0:
            forecast_values = forecast_data['forecasted_sales']
            std_dev = np.std(forecast_values) * 0.15  # 15% confidence band
            
            upper_bound = [val + std_dev for val in forecast_values]
            lower_bound = [max(0, val - std_dev) for val in forecast_values]  # Ensure non-negative
            
            # Upper confidence bound
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=upper_bound,
                    mode='lines',
                    name='Upper Confidence',
                    line=dict(color='rgba(231, 76, 60, 0.3)', width=1, dash='dot'),
                    showlegend=True,
                    hoverinfo='skip'
                )
            )
            
            # Lower confidence bound
            fig.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=lower_bound,
                    mode='lines',
                    name='Lower Confidence',
                    line=dict(color='rgba(231, 76, 60, 0.3)', width=1, dash='dot'),
                    fill='tonexty',
                    fillcolor='rgba(231, 76, 60, 0.15)',
                    showlegend=True,
                    hoverinfo='skip'
                )
            )
        
        # Add forecast start line with label
        if historical_data and hist_dates is not None and len(hist_dates) > 0:
            # Convert pandas Timestamp to string for Plotly compatibility
            forecast_start = pd.Timestamp(forecast_dates[0]).strftime('%Y-%m-%d')
            
            # Add vertical line using add_shape instead of add_vline
            fig.add_shape(
                type="line",
                x0=forecast_start,
                x1=forecast_start,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="#2ecc71", width=2, dash="solid")
            )
            
            # Add annotation separately
            fig.add_annotation(
                x=forecast_start,
                y=1,
                yref="paper",
                text="← Historical | Forecast →",
                showarrow=False,
                font=dict(size=12, color="#2ecc71"),
                bgcolor="rgba(46, 204, 113, 0.2)",
                borderpad=4,
                yshift=10
            )
            
            # Connect last historical to first forecast with smooth transition
            last_hist_date = hist_dates[-1]
            last_hist_sales = historical_data['sales'][-1]
            first_forecast_date = forecast_dates[0]
            first_forecast_sales = forecast_data['forecasted_sales'][0]
            
            fig.add_trace(
                go.Scatter(
                    x=[last_hist_date, first_forecast_date],
                    y=[last_hist_sales, first_forecast_sales],
                    mode='lines',
                    name='Transition',
                    line=dict(color='#95a5a6', width=2, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip'
                )
            )
    
    # Update layout with better styling
    fig.update_layout(
        height=600,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#CCCCCC",
            borderwidth=1
        ),
        xaxis_title="<b>Date</b>",
        yaxis_title="<b>Sales Units</b>",
        template="plotly_white",
        plot_bgcolor='rgba(240, 240, 240, 0.5)',
        font=dict(size=12),
        margin=dict(l=60, r=40, t=80, b=60)
    )
    
    # Update axes
    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        showline=True,
        linewidth=2,
        linecolor='gray'
    )
    
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(200, 200, 200, 0.3)',
        showline=True,
        linewidth=2,
        linecolor='gray',
        rangemode='tozero'
    )
    
    return fig

# Main App
def main():
    # Header
    st.markdown('<h1 class="main-header">📈 Sales Forecasting Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Check API health
    try:
        health = requests.get(f"{API_URL}/health").json()
        if health['status'] != 'healthy':
            st.error("⚠️ API is not healthy. Please check the backend service.")
            return
    except:
        st.error("❌ Cannot connect to API. Please ensure the backend is running.")
        st.info(f"Expected API URL: {API_URL}")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Forecast Parameters")
        
        # Fetch stores
        stores = fetch_stores()
        if not stores:
            st.error("No stores available")
            return
        
        # Store selection
        selected_store = st.selectbox(
            "Select Store",
            options=stores,
            format_func=lambda x: f"Store {x}"
        )
        
        # Fetch items for selected store
        items = fetch_items(selected_store)
        if not items:
            st.error(f"No items available for Store {selected_store}")
            return
        
        # Item selection
        selected_item = st.selectbox(
            "Select Item",
            options=items,
            format_func=lambda x: f"Item {x}"
        )
        
        # Historical days - default to 90 (3 months)
        historical_days = st.slider(
            "Historical Days to Display",
            min_value=30,
            max_value=180,
            value=90,
            step=30,
            help="Show past sales data for context (90 days = 3 months)"
        )
        
        # Generate forecast button
        generate_button = st.button("🔮 Generate Forecast", type="primary", use_container_width=True)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("""
        This dashboard provides 30-day sales forecasts using deep learning models.
        
        **Features:**
        - Historical sales trends (3 months)
        - 30-day ahead forecasts
        - Statistical summaries
        - Interactive visualizations
        """)
    
    # Main content
    if generate_button:
        with st.spinner("🔄 Generating forecast..."):
            # Fetch data - default to 90 days (3 months) for historical
            forecast_data = fetch_forecast(selected_store, selected_item)
            historical_data = fetch_historical(selected_store, selected_item, historical_days)
            
            if forecast_data:
                # Display metrics
                st.subheader(f"📊 Forecast Summary - Store {selected_store}, Item {selected_item}")
                st.caption(f"Historical Data: Last {historical_days} days (~{historical_days//30} months) | Forecast: Next 30 days")
                
                col1, col2, col3, col4 = st.columns(4)
                
                # Calculate trend if we have historical data
                trend_delta = None
                if historical_data and len(historical_data['sales']) > 0:
                    hist_avg = np.mean(historical_data['sales'])
                    forecast_avg = forecast_data['avg_sales']
                    trend_delta = ((forecast_avg - hist_avg) / hist_avg) * 100
                
                with col1:
                    st.metric(
                        label="📈 Total 30-Day Sales",
                        value=f"{forecast_data['total_sales']:.0f}",
                        delta=None
                    )
                
                with col2:
                    st.metric(
                        label="📊 Average Daily Sales",
                        value=f"{forecast_data['avg_sales']:.2f}",
                        delta=f"{trend_delta:+.1f}% vs history" if trend_delta is not None else None,
                        delta_color="normal"
                    )
                
                with col3:
                    st.metric(
                        label="⬇️ Minimum Daily Sales",
                        value=f"{forecast_data['min_sales']:.2f}",
                        delta=None
                    )
                
                with col4:
                    st.metric(
                        label="⬆️ Maximum Daily Sales",
                        value=f"{forecast_data['max_sales']:.2f}",
                        delta=None
                    )
                
                st.markdown("---")
                
                # Comparison metrics
                if historical_data and len(historical_data['sales']) > 0:
                    st.subheader("📊 Historical vs Forecast Comparison")
                    
                    comp_col1, comp_col2, comp_col3 = st.columns(3)
                    
                    hist_avg = np.mean(historical_data['sales'])
                    hist_total = np.sum(historical_data['sales'])
                    hist_std = np.std(historical_data['sales'])
                    
                    with comp_col1:
                        st.metric(
                            "Historical Avg (Daily)",
                            f"{hist_avg:.2f}",
                            delta=None
                        )
                        st.metric(
                            "Forecast Avg (Daily)",
                            f"{forecast_data['avg_sales']:.2f}",
                            delta=f"{((forecast_data['avg_sales']-hist_avg)/hist_avg)*100:+.1f}%"
                        )
                    
                    with comp_col2:
                        days_ratio = len(historical_data['sales']) / 30
                        hist_30day_equiv = hist_total / days_ratio
                        st.metric(
                            f"Historical Total ({len(historical_data['sales'])} days)",
                            f"{hist_total:.0f}"
                        )
                        st.metric(
                            "Forecast Total (30 days)",
                            f"{forecast_data['total_sales']:.0f}",
                            delta=f"{((forecast_data['total_sales']-hist_30day_equiv)/hist_30day_equiv)*100:+.1f}%"
                        )
                    
                    with comp_col3:
                        forecast_std = np.std(forecast_data['forecasted_sales'])
                        st.metric(
                            "Historical Volatility (σ)",
                            f"{hist_std:.2f}"
                        )
                        st.metric(
                            "Forecast Volatility (σ)",
                            f"{forecast_std:.2f}",
                            delta=f"{((forecast_std-hist_std)/hist_std)*100:+.1f}%"
                        )
                
                st.markdown("---")
                
                # Main forecast plot
                st.subheader("📈 Historical Trends & 30-Day Forecast")
                
                # Add info about the data being shown
                if historical_data:
                    hist_start = pd.to_datetime(historical_data['dates'][0]).strftime('%b %d, %Y')
                    hist_end = pd.to_datetime(historical_data['dates'][-1]).strftime('%b %d, %Y')
                    forecast_start = pd.to_datetime(forecast_data['dates'][0]).strftime('%b %d, %Y')
                    forecast_end = pd.to_datetime(forecast_data['dates'][-1]).strftime('%b %d, %Y')
                    
                    st.info(f"""
                    **📅 Data Period:**
                    - Historical: {hist_start} to {hist_end} ({len(historical_data['dates'])} days, ~{len(historical_data['dates'])//30} months)
                    - Forecast: {forecast_start} to {forecast_end} (30 days)
                    """)
                
                forecast_plot = create_forecast_plot(historical_data, forecast_data)
                st.plotly_chart(forecast_plot, use_container_width=True)
                
                st.markdown("---")
                
                # Statistics plots
                st.subheader("📊 Detailed Forecast Analysis")
                
                # Create two columns for side-by-side charts
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("#### 📅 Daily Forecast Trend")
                    # Daily forecast line chart
                    daily_fig = go.Figure()
                    daily_fig.add_trace(
                        go.Scatter(
                            x=list(range(1, len(forecast_data['forecasted_sales']) + 1)),
                            y=forecast_data['forecasted_sales'],
                            mode='lines+markers',
                            name='Daily Forecast',
                            line=dict(color='#e74c3c', width=2),
                            marker=dict(size=6),
                            fill='tozeroy',
                            fillcolor='rgba(231, 76, 60, 0.2)',
                            hovertemplate='<b>Day %{x}</b><br>Sales: %{y:.2f}<extra></extra>'
                        )
                    )
                    daily_fig.update_layout(
                        height=350,
                        xaxis_title="Day Number",
                        yaxis_title="Forecasted Sales",
                        template="plotly_white",
                        showlegend=False
                    )
                    st.plotly_chart(daily_fig, use_container_width=True)
                
                with col_right:
                    st.markdown("#### 📦 Distribution Analysis")
                    # Box plot
                    box_fig = go.Figure()
                    box_fig.add_trace(
                        go.Box(
                            y=forecast_data['forecasted_sales'],
                            name='Forecast',
                            marker_color='#e74c3c',
                            boxmean='sd',
                            hovertemplate='<b>Statistics</b><br>Value: %{y:.2f}<extra></extra>'
                        )
                    )
                    box_fig.update_layout(
                        height=350,
                        yaxis_title="Forecasted Sales",
                        template="plotly_white",
                        showlegend=False
                    )
                    st.plotly_chart(box_fig, use_container_width=True)
                
                # Add histogram
                st.markdown("#### 📊 Sales Distribution Histogram")
                hist_fig = go.Figure()
                hist_fig.add_trace(
                    go.Histogram(
                        x=forecast_data['forecasted_sales'],
                        nbinsx=15,
                        name='Distribution',
                        marker_color='#3498db',
                        hovertemplate='<b>Range</b>: %{x}<br><b>Count</b>: %{y}<extra></extra>'
                    )
                )
                hist_fig.update_layout(
                    height=300,
                    xaxis_title="Sales Range",
                    yaxis_title="Frequency",
                    template="plotly_white",
                    showlegend=False
                )
                st.plotly_chart(hist_fig, use_container_width=True)
                
                st.markdown("---")
                
                # Data table
                st.subheader("📋 Detailed Forecast Data Table")
                
                # Add trend analysis
                forecast_vals = forecast_data['forecasted_sales']
                first_week_avg = np.mean(forecast_vals[:7])
                last_week_avg = np.mean(forecast_vals[-7:])
                trend_direction = "📈 Increasing" if last_week_avg > first_week_avg else "📉 Decreasing"
                trend_pct = ((last_week_avg - first_week_avg) / first_week_avg) * 100
                
                trend_col1, trend_col2, trend_col3 = st.columns([1, 1, 2])
                with trend_col1:
                    st.metric("First Week Avg", f"{first_week_avg:.2f}")
                with trend_col2:
                    st.metric("Last Week Avg", f"{last_week_avg:.2f}", delta=f"{trend_pct:+.1f}%")
                with trend_col3:
                    st.info(f"**Trend:** {trend_direction} over the forecast period")
                
                # Create DataFrame
                forecast_df = pd.DataFrame({
                    'Date': forecast_data['dates'],
                    'Forecasted Sales': forecast_data['forecasted_sales'],
                    'Day': range(1, len(forecast_data['dates']) + 1)
                })
                
                # Add week info
                forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
                forecast_df['Day of Week'] = forecast_df['Date'].dt.day_name()
                forecast_df['Week'] = forecast_df['Date'].dt.isocalendar().week
                
                # Format for display
                display_df = forecast_df[['Day', 'Date', 'Day of Week', 'Week', 'Forecasted Sales']].copy()
                display_df['Date'] = display_df['Date'].dt.strftime('%Y-%m-%d')
                display_df['Forecasted Sales'] = display_df['Forecasted Sales'].round(2)
                
                st.dataframe(display_df, use_container_width=True, height=400)
                
                # Download button
                csv = display_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Forecast Data (CSV)",
                    data=csv,
                    file_name=f"forecast_store_{selected_store}_item_{selected_item}.csv",
                    mime="text/csv"
                )
                
                # Weekly summary
                st.markdown("---")
                st.subheader("📅 Weekly Summary")
                
                weekly_summary = forecast_df.groupby('Week').agg({
                    'Forecasted Sales': ['sum', 'mean', 'std', 'min', 'max']
                }).round(2)
                weekly_summary.columns = ['Total Sales', 'Avg Sales', 'Std Dev', 'Min Sales', 'Max Sales']
                weekly_summary = weekly_summary.reset_index()
                
                st.dataframe(weekly_summary, use_container_width=True)
    else:
        # Welcome message
        st.info("👈 Please select a store and item from the sidebar, then click 'Generate Forecast' to view predictions.")
        
        # Display sample image or instructions
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            ### 🚀 Getting Started
            
            1. **Select a Store** from the dropdown menu
            2. **Select an Item** for that store
            3. **Adjust** the historical days slider (default: 90 days = 3 months)
            4. **Click** the 'Generate Forecast' button
            5. **View** the interactive forecast showing:
               - Last 3 months of historical sales
               - Next 30 days of predicted sales
            6. **Download** the forecast data as CSV
            
            ### 📊 What You'll See
            
            - **Metrics**: Total, average, min, and max sales
            - **Combined Chart**: 3 months history + 30-day forecast
            - **Visualizations**: Interactive line charts and distributions
            - **Data Table**: Complete forecast breakdown by day
            - **Weekly Summary**: Aggregated weekly statistics
            """)

if __name__ == "__main__":
    main()