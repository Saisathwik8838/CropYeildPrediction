"""
Visualization Module for AgriClimate Insight Portal
Creates interactive Plotly visualizations for climate and agriculture data
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class DataVisualizer:
    """
    Main class for creating interactive visualizations
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize the visualizer with a DataFrame
        
        Args:
            df: Preprocessed DataFrame
        """
        self.df = df.copy()
        
        # Premium color palette with semantic meaning
        self.colors = {
            # Primary brand colors
            'primary': '#0ea5e9',
            'primary-light': '#38bdf8',
            'primary-dark': '#0284c7',
            
            # Semantic colors for data types
            'rainfall': '#3b82f6',      # Blue for rainfall
            'rainfall-light': '#60a5fa',
            'rainfall-dark': '#1d4ed8',
            'temperature': '#ef4444',   # Red for temperature
            'temperature-light': '#f87171',
            'temperature-dark': '#b91c1c',
            'crop': '#22c55e',          # Green for crops/yield
            'crop-light': '#4ade80',
            'crop-dark': '#16a34a',
            
            # Supporting colors
            'success': '#22c55e',
            'warning': '#f59e0b',
            'danger': '#ef4444',
            'info': '#06b6d4',
            'neutral': '#64748b',
            'neutral-light': '#94a3b8',
            'neutral-dark': '#334155',
            
            # Chart palettes
            'palette_rainfall': ['#eff6ff', '#dbeafe', '#bfdbfe', '#93c5fd', '#60a5fa', '#3b82f6', '#2563eb', '#1d4ed8'],
            'palette_temperature': ['#fef2f2', '#fee2e2', '#fecaca', '#fca5a5', '#f87171', '#ef4444', '#dc2626', '#b91c1c'],
            'palette_crop': ['#f0fdf4', '#dcfce7', '#bbf7d0', '#86efac', '#4ade80', '#22c55e', '#16a34a', '#15803d'],
            'palette_diverging': ['#7c3aed', '#a855f7', '#c084fc', '#e879f9', '#fbbf24', '#f59e0b', '#d97706', '#b45309']
        }
        
        # Premium chart theme
        self.chart_theme = {
            'layout': {
                'font': {'family': 'Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif', 'color': '#1e293b'},
                'paper_bgcolor': 'rgba(255,255,255,0.9)',
                'plot_bgcolor': 'rgba(248,250,252,0.5)',
                'margin': {'l': 60, 'r': 60, 't': 80, 'b': 60},
                'showlegend': True,
                'legend': {
                    'orientation': 'h',
                    'yanchor': 'bottom',
                    'y': 1.02,
                    'xanchor': 'right',
                    'x': 1,
                    'bgcolor': 'rgba(255,255,255,0.8)',
                    'bordercolor': 'rgba(0,0,0,0.1)',
                    'borderwidth': 1,
                    'font': {'color': '#1e293b'}
                }
            },
            'xaxis': {
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.05)',
                'gridwidth': 1,
                'showline': True,
                'linecolor': 'rgba(0,0,0,0.1)',
                'linewidth': 1,
                'tickfont': {'size': 12, 'color': '#1e293b'},
                'title': {'font': {'size': 14, 'color': '#1e293b', 'family': 'Inter'}}
            },
            'yaxis': {
                'showgrid': True,
                'gridcolor': 'rgba(0,0,0,0.05)',
                'gridwidth': 1,
                'showline': True,
                'linecolor': 'rgba(0,0,0,0.1)',
                'linewidth': 1,
                'tickfont': {'size': 12, 'color': '#1e293b'},
                'title': {'font': {'size': 14, 'color': '#1e293b', 'family': 'Inter'}}
            }
        }
        
    def create_line_chart(self, x_col: str, y_col: str, group_col: str = None, 
                         title: str = None, y_title: str = None) -> go.Figure:
        """
        Create premium interactive line chart with animations and semantic colors
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            group_col: Column for grouping/color
            title: Chart title
            y_title: Y-axis title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Determine semantic color based on y_col
        semantic_color = self._get_semantic_color(y_col)
        
        if group_col and group_col in self.df.columns:
            # Grouped line chart with semantic colors
            groups = self.df[group_col].unique()
            color_palette = self._get_color_palette(len(groups), semantic_color)
            
            for i, group in enumerate(groups):
                group_data = self.df[self.df[group_col] == group]
                fig.add_trace(go.Scatter(
                    x=group_data[x_col],
                    y=group_data[y_col],
                    mode='lines+markers',
                    name=str(group),
                    line=dict(
                        width=3,
                        color=color_palette[i],
                        shape='spline',
                        smoothing=0.3
                    ),
                    marker=dict(
                        size=8,
                        color=color_palette[i],
                        line=dict(width=2, color='white'),
                        opacity=0.8
                    ),
                    hovertemplate=f'<b>{group}</b><br>' +
                                 f'{x_col}: %{{x}}<br>' +
                                 f'{y_col}: %{{y}}<br>' +
                                 '<extra></extra>',
                    showlegend=True
                ))
        else:
            # Single line chart with semantic color
            fig.add_trace(go.Scatter(
                x=self.df[x_col],
                y=self.df[y_col],
                mode='lines+markers',
                name=y_col,
                line=dict(
                    color=semantic_color,
                    width=4,
                    shape='spline',
                    smoothing=0.3
                ),
                marker=dict(
                    size=10,
                    color=semantic_color,
                    line=dict(width=2, color='white'),
                    opacity=0.8
                ),
                hovertemplate=f'<b>{y_col}</b><br>' +
                             f'{x_col}: %{{x}}<br>' +
                             f'{y_col}: %{{y}}<br>' +
                             '<extra></extra>',
                showlegend=True
            ))
        
        # Apply premium styling
        fig.update_layout(
            title={
                'text': title or f"{y_col} Trends",
                'font': {'size': 20, 'color': '#1e293b', 'family': 'Inter'},
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title=x_col,
            yaxis_title=y_title or y_col,
            hovermode='x unified',
            height=500,
            **self.chart_theme['layout'],
            xaxis=self.chart_theme['xaxis'],
            yaxis=self.chart_theme['yaxis'],
            # Add subtle animation
            transition={'duration': 500},
            # Enhanced hover
            hoverlabel={
                'bgcolor': 'rgba(255,255,255,0.95)',
                'bordercolor': 'rgba(0,0,0,0.2)',
                'font': {'family': 'Inter', 'size': 12, 'color': '#1e293b'}
            }
        )
        
        return fig
    
    def _get_semantic_color(self, column_name: str) -> str:
        """
        Get semantic color based on column name
        
        Args:
            column_name: Name of the column
            
        Returns:
            Hex color code
        """
        column_lower = column_name.lower()
        
        if any(word in column_lower for word in ['rainfall', 'precipitation', 'water']):
            return self.colors['rainfall']
        elif any(word in column_lower for word in ['temperature', 'temp', 'heat']):
            return self.colors['temperature']
        elif any(word in column_lower for word in ['yield', 'crop', 'production', 'harvest']):
            return self.colors['crop']
        else:
            return self.colors['primary']
    
    def _get_color_palette(self, num_colors: int, base_color: str) -> List[str]:
        """
        Generate color palette for multiple series
        
        Args:
            num_colors: Number of colors needed
            base_color: Base color to generate palette from
            
        Returns:
            List of hex color codes
        """
        if base_color == self.colors['rainfall']:
            palette = self.colors['palette_rainfall']
        elif base_color == self.colors['temperature']:
            palette = self.colors['palette_temperature']
        elif base_color == self.colors['crop']:
            palette = self.colors['palette_crop']
        else:
            palette = self.colors['palette_diverging']
        
        # Return the required number of colors
        return palette[:num_colors] if num_colors <= len(palette) else palette
    
    def create_correlation_heatmap(self, columns: List[str] = None, 
                                  title: str = "Correlation Matrix") -> go.Figure:
        """
        Create premium correlation heatmap with enhanced styling
        
        Args:
            columns: List of columns to include in correlation
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = self.df[columns].corr()
        
        # Create premium heatmap with custom colorscale
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale=[
                [0.0, '#7c3aed'],    # Purple for negative correlations
                [0.5, '#ffffff'],    # White for zero correlation
                [1.0, '#22c55e']     # Green for positive correlations
            ],
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="<b>%{text}</b>",
            textfont={"size": 11, "color": "white"},
            hoverongaps=False,
            hovertemplate='<b>%{y} vs %{x}</b><br>' +
                         'Correlation: %{z:.3f}<br>' +
                         '<extra></extra>',
            showscale=True,
            colorbar=dict(
                title="Correlation",
                titleside="right",
                tickmode="linear",
                tick0=-1,
                dtick=0.5,
                len=0.8,
                y=0.5,
                yanchor="middle"
            )
        ))
        
        # Apply premium styling
        fig.update_layout(
            title={
                'text': title,
                'font': {'size': 20, 'color': '#1e293b', 'family': 'Inter'},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=600,
            width=600,
            **self.chart_theme['layout'],
            xaxis={
                **self.chart_theme['xaxis'],
                'side': 'bottom',
                'tickangle': -45
            },
            yaxis={
                **self.chart_theme['yaxis'],
                'autorange': 'reversed'
            },
            hoverlabel={
                'bgcolor': 'rgba(255,255,255,0.95)',
                'bordercolor': 'rgba(0,0,0,0.2)',
                'font': {'family': 'Inter', 'size': 12, 'color': '#1e293b'}
            }
        )
        
        return fig
    
    def create_scatter_plot(self, x_col: str, y_col: str, color_col: str = None,
                           size_col: str = None, title: str = None) -> go.Figure:
        """
        Create interactive scatter plot
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            color_col: Column for color mapping
            size_col: Column for size mapping
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = px.scatter(
            self.df,
            x=x_col,
            y=y_col,
            color=color_col,
            size=size_col,
            title=title or f"{y_col} vs {x_col}",
            hover_data=[col for col in self.df.columns if col not in [x_col, y_col, color_col, size_col]][:3]
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_bar_chart(self, x_col: str, y_col: str, group_col: str = None,
                        title: str = None, orientation: str = 'v') -> go.Figure:
        """
        Create interactive bar chart
        
        Args:
            x_col: Column for x-axis
            y_col: Column for y-axis
            group_col: Column for grouping
            title: Chart title
            orientation: 'v' for vertical, 'h' for horizontal
            
        Returns:
            Plotly figure object
        """
        if group_col and group_col in self.df.columns:
            fig = px.bar(
                self.df,
                x=x_col,
                y=y_col,
                color=group_col,
                title=title or f"{y_col} by {x_col}",
                barmode='group'
            )
        else:
            fig = px.bar(
                self.df,
                x=x_col,
                y=y_col,
                title=title or f"{y_col} by {x_col}"
            )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_box_plot(self, x_col: str, y_col: str, title: str = None) -> go.Figure:
        """
        Create interactive box plot
        
        Args:
            x_col: Column for x-axis (categorical)
            y_col: Column for y-axis (numerical)
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = px.box(
            self.df,
            x=x_col,
            y=y_col,
            title=title or f"{y_col} Distribution by {x_col}",
            points="all"
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_histogram(self, col: str, bins: int = 30, title: str = None) -> go.Figure:
        """
        Create premium interactive histogram with semantic colors
        
        Args:
            col: Column to plot
            bins: Number of bins
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Get semantic color for the column
        semantic_color = self._get_semantic_color(col)
        
        fig = px.histogram(
            self.df,
            x=col,
            nbins=bins,
            title=title or f"Distribution of {col}",
            marginal="box",
            color_discrete_sequence=[semantic_color],
            opacity=0.8
        )
        
        # Enhance the histogram bars
        fig.update_traces(
            marker=dict(
                line=dict(width=1, color='white'),
                opacity=0.8
            ),
            hovertemplate=f'<b>{col}</b><br>' +
                         'Range: %{x}<br>' +
                         'Count: %{y}<br>' +
                         '<extra></extra>'
        )
        
        # Apply premium styling
        fig.update_layout(
            title={
                'text': title or f"Distribution of {col}",
                'font': {'size': 20, 'color': '#1e293b', 'family': 'Inter'},
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            **self.chart_theme['layout'],
            xaxis=self.chart_theme['xaxis'],
            yaxis=self.chart_theme['yaxis'],
            hoverlabel={
                'bgcolor': 'rgba(255,255,255,0.95)',
                'bordercolor': 'rgba(0,0,0,0.2)',
                'font': {'family': 'Inter', 'size': 12, 'color': '#1e293b'}
            },
            # Add subtle animation
            transition={'duration': 500}
        )
        
        return fig
    
    def create_time_series_plot(self, date_col: str, value_cols: List[str],
                               title: str = "Time Series Analysis") -> go.Figure:
        """
        Create multi-line time series plot
        
        Args:
            date_col: Date column
            value_cols: List of value columns to plot
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, col in enumerate(value_cols):
            if col in self.df.columns:
                fig.add_trace(go.Scatter(
                    x=self.df[date_col],
                    y=self.df[col],
                    mode='lines+markers',
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2),
                    marker=dict(size=6)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_prediction_plot(self, actual: np.ndarray, predicted: np.ndarray,
                              title: str = "Actual vs Predicted") -> go.Figure:
        """
        Create actual vs predicted scatter plot
        
        Args:
            actual: Actual values
            predicted: Predicted values
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add scatter plot
        fig.add_trace(go.Scatter(
            x=actual,
            y=predicted,
            mode='markers',
            name='Predictions',
            marker=dict(
                color=self.colors['primary'],
                size=8,
                opacity=0.7
            )
        ))
        
        # Add perfect prediction line
        min_val = min(min(actual), min(predicted))
        max_val = max(max(actual), max(predicted))
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color=self.colors['danger'], dash='dash', width=2)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Actual Values",
            yaxis_title="Predicted Values",
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_feature_importance_plot(self, importance_dict: Dict[str, float],
                                     title: str = "Feature Importance") -> go.Figure:
        """
        Create feature importance bar chart
        
        Args:
            importance_dict: Dictionary of feature names and importance values
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        features = list(importance_dict.keys())
        importance_values = list(importance_dict.values())
        
        # Sort by importance
        sorted_data = sorted(zip(features, importance_values), key=lambda x: x[1], reverse=True)
        features, importance_values = zip(*sorted_data)
        
        fig = go.Figure(data=[
            go.Bar(
                x=importance_values,
                y=features,
                orientation='h',
                marker=dict(color=self.colors['success'])
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title="Importance",
            yaxis_title="Features",
            template='plotly_white',
            height=max(400, len(features) * 30)
        )
        
        return fig
    
    def create_geographic_plot(self, location_col: str, value_col: str,
                              title: str = "Geographic Distribution") -> go.Figure:
        """
        Create geographic distribution plot (simplified for Indian states)
        
        Args:
            location_col: Column containing location names
            value_col: Column containing values to plot
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        # Aggregate data by location
        geo_data = self.df.groupby(location_col)[value_col].mean().reset_index()
        
        fig = px.bar(
            geo_data,
            x=location_col,
            y=value_col,
            title=title,
            color=value_col,
            color_continuous_scale='Viridis'
        )
        
        fig.update_layout(
            template='plotly_white',
            height=500,
            xaxis_tickangle=-45
        )
        
        return fig
    
    def create_subplot_dashboard(self, plots_config: List[Dict[str, Any]],
                                title: str = "Dashboard") -> go.Figure:
        """
        Create a dashboard with multiple subplots
        
        Args:
            plots_config: List of plot configurations
            title: Dashboard title
            
        Returns:
            Plotly figure object
        """
        rows = len(plots_config)
        cols = 1
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[config.get('title', '') for config in plots_config],
            vertical_spacing=0.1
        )
        
        for i, config in enumerate(plots_config):
            plot_type = config.get('type', 'scatter')
            
            if plot_type == 'scatter':
                fig.add_trace(
                    go.Scatter(
                        x=self.df[config['x']],
                        y=self.df[config['y']],
                        mode='markers',
                        name=config.get('name', ''),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
            elif plot_type == 'bar':
                fig.add_trace(
                    go.Bar(
                        x=self.df[config['x']],
                        y=self.df[config['y']],
                        name=config.get('name', ''),
                        showlegend=False
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=300 * rows
        )
        
        return fig
    
    def create_anomaly_plot(self, date_col: str, value_col: str,
                           anomaly_data: pd.DataFrame,
                           title: str = "Anomaly Detection") -> go.Figure:
        """
        Create plot highlighting anomalies
        
        Args:
            date_col: Date column
            value_col: Value column
            anomaly_data: DataFrame containing anomaly points
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add normal data points
        normal_data = self.df[~self.df.index.isin(anomaly_data.index)]
        fig.add_trace(go.Scatter(
            x=normal_data[date_col],
            y=normal_data[value_col],
            mode='markers',
            name='Normal',
            marker=dict(color=self.colors['primary'], size=6)
        ))
        
        # Add anomaly points
        fig.add_trace(go.Scatter(
            x=anomaly_data[date_col],
            y=anomaly_data[value_col],
            mode='markers',
            name='Anomaly',
            marker=dict(color=self.colors['danger'], size=10, symbol='x')
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=date_col,
            yaxis_title=value_col,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def create_trend_analysis_plot(self, date_col: str, value_col: str,
                                  trend_data: pd.Series,
                                  title: str = "Trend Analysis") -> go.Figure:
        """
        Create plot showing trend analysis
        
        Args:
            date_col: Date column
            value_col: Value column
            trend_data: Trend data (rolling mean)
            title: Chart title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add original data
        fig.add_trace(go.Scatter(
            x=self.df[date_col],
            y=self.df[value_col],
            mode='markers',
            name='Original Data',
            marker=dict(color=self.colors['primary'], size=4, opacity=0.6)
        ))
        
        # Add trend line
        fig.add_trace(go.Scatter(
            x=trend_data.index,
            y=trend_data.values,
            mode='lines',
            name='Trend',
            line=dict(color=self.colors['danger'], width=3)
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title=date_col,
            yaxis_title=value_col,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    def get_visualization_summary(self) -> Dict[str, str]:
        """
        Generate summary of available visualizations
        
        Returns:
            Dictionary with visualization insights
        """
        summary = {}
        
        # Data shape
        summary['data_shape'] = f"{self.df.shape[0]} rows, {self.df.shape[1]} columns"
        
        # Available columns
        numerical_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = self.df.select_dtypes(include=['object']).columns.tolist()
        
        summary['numerical_columns'] = f"{len(numerical_cols)} numerical columns"
        summary['categorical_columns'] = f"{len(categorical_cols)} categorical columns"
        
        # Date columns
        date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'year' in col.lower()]
        summary['date_columns'] = f"{len(date_cols)} date/time columns"
        
        return summary


if __name__ == "__main__":
    # Test the visualizer
    from data_preprocessing import DataPreprocessor
    
    # Load and preprocess data
    preprocessor = DataPreprocessor()
    processed_data = preprocessor.preprocess_all()
    
    # Create visualizer
    visualizer = DataVisualizer(processed_data)
    
    # Test some visualizations
    if 'Year' in processed_data.columns and 'Yield' in processed_data.columns:
        line_fig = visualizer.create_line_chart('Year', 'Yield', 'State', 'Yield Trends by State')
        print("✅ Line chart created")
    
    if len(processed_data.select_dtypes(include=[np.number]).columns) > 1:
        corr_fig = visualizer.create_correlation_heatmap()
        print("✅ Correlation heatmap created")
    
    # Print summary
    summary = visualizer.get_visualization_summary()
    print("\n📊 Visualization Summary:")
    for key, value in summary.items():
        print(f"   {key}: {value}")
