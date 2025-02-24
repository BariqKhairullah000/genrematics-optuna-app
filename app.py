import streamlit as st
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime
import gc
import logging
import sys
import codecs
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
import psutil
import os
import time

# Fix console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Constants and Configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / 'models'
LOG_DIR = BASE_DIR / 'logs'
DATA_DIR = BASE_DIR / 'data'
BACKUP_DIR = BASE_DIR / 'backups'

class Config:
    # Model Parameters
    MODEL_PARAMS = {
        'EPOCHS': 20,
        'BATCH_SIZE': 10,
        'LEARNING_RATE': 1e-5,
        'MAX_LENGTH': 512,
        'TEST_SIZE': 0.15,
        'WEIGHT_DECAY': 0.05,
        'MIXUP_PROB': 0.5,
        'PATIENCE': 5,
        'SMOOTHING': 0.2
    }


    OPTIM_PARAMS = {
        'BATCH_SIZE': [8, 16, 32],
        'LEARNING_RATE': [1e-5, 2e-5, 3e-5],
        'WEIGHT_DECAY': [0.01, 0.02],
        'MIXUP_PROB': [0.2, 0.3],
        'SMOOTHING': [0.1, 0.15]
    }
    
    # Device Configuration
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAMPLE_SIZE: Optional[int] = None

def local_css():
    st.markdown("""
        <style>
        /* Dark theme colors */
        :root {
            --background-dark: #1a1a1a;
            --card-bg: #2d2d2d;
            --text-primary: #ffffff;
            --text-secondary: #b3b3b3;
            --accent-blue: #4ECDC4;
            --accent-red: #FF6B6B;
            --accent-yellow: #FFE66D;
        }

        /* Main container styling */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            background-color: var(--background-dark);
        }

        /* Header styling */
        h1 {
            color: var(--text-primary) !important;
            font-size: 1.875rem !important;
            font-weight: 700 !important;
            margin-bottom: 0.5rem !important;
        }

        /* Card styling */
        .card {
            background-color: var(--card-bg);
            border-radius: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
            margin-bottom: 1rem;
            padding: 1.5rem;
        }

        .card-title {
            color: var(--accent-blue);
            font-size: 1.25rem;
            font-weight: 600;
            margin-bottom: 1rem;
        }

        /* Log viewer */
        .log-viewer {
            background-color: var(--card-bg);
            border-radius: 0.5rem;
            padding: 1rem;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }

        /* Confidence sections */
        .confidence-section {
            margin-bottom: 1rem;
            padding: 1rem;
            border-radius: 0.5rem;
            background: var(--card-bg);
        }
        </style>
    """, unsafe_allow_html=True)

class DataProcessor:
    """Class for handling data preprocessing and loading"""
    @staticmethod
    def clean_text(text: str) -> str:
        """Clean and preprocess text data"""
        if isinstance(text, str):
            text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
            text = re.sub(r'\S+@\S+', '', text)
            text = re.sub(r'[^\w\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text)
            return text.strip().lower()
        return ''

    @staticmethod
    def load_and_preprocess_data(data_path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load and preprocess data with proper encoding handling"""
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        encodings_to_try = ['utf-8', 'utf-8-sig', 'latin1', 'iso-8859-1', 'cp1252']
        df = None

        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(data_path, encoding=encoding)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if df is None:
            raise UnicodeError(f"Failed to read file with any of these encodings: {encodings_to_try}")

        if sample_size:
            df = df.head(sample_size)

        df['sinopsis'] = df['sinopsis'].apply(DataProcessor.clean_text)
        df['genre'] = df['genre'].str.split(',')
        df = df.dropna(subset=['sinopsis', 'genre'])

        return df

def get_memory_usage() -> float:
    """Get current memory usage of the program"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # in MB

def log_memory(step_name: str) -> None:
    """Log memory usage with consistent format"""
    memory = get_memory_usage()
    logging.info(f"Memory usage after {step_name}: {memory:.2f} MB")

def get_all_experiments():
    experiments_dir = LOG_DIR / 'experiments'
    if not experiments_dir.exists():
        raise FileNotFoundError("No experiments directory found")
    
    experiments = [d for d in experiments_dir.iterdir() if d.is_dir()]
    if not experiments:
        raise FileNotFoundError("No experiments found")
    
    sorted_experiments = sorted(
        experiments,
        key=lambda x: datetime.strptime(x.name, "%Y%m%d_%H%M%S"),
        reverse=True
    )
    
    return [(datetime.strptime(exp.name, "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M:%S"), exp)
            for exp in sorted_experiments]

@st.cache_resource
def load_model_and_history(model_path, tokenizer_path, metrics_dir, experiment_path):
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = model.to(device)
        model.eval()
        
        with open(metrics_dir / 'training_history.json', 'r', encoding='utf-8') as f:
            training_history = json.load(f)
        
        with open(metrics_dir / 'evaluation_metrics.json', 'r', encoding='utf-8') as f:
            evaluation_metrics = json.load(f)
            
        with open(experiment_path / 'final_configuration.json', 'r', encoding='utf-8') as f:
            experiment_info = json.load(f)
            
        return model, tokenizer, training_history, evaluation_metrics, experiment_info, device
    except Exception as e:
        st.error(f"Error loading model and data: {str(e)}")
        st.stop()

def predict_genre(text, model, tokenizer, device, metrics_dir):
    text = DataProcessor.clean_text(text)
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        padding='max_length',
        truncation=True
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits)
    
    with open(metrics_dir / 'training_history.json', 'r', encoding='utf-8') as f:
        history = json.load(f)
        genres = history['model_info']['classes']
    
    predictions = []
    for idx, prob in enumerate(probs[0]):
        predictions.append({
            'genre': genres[idx],
            'probability': float(prob)
        })
    
    return sorted(predictions, key=lambda x: x['probability'], reverse=True)

def load_dataset_info():
    """Load and process dataset information"""
    try:
        df = pd.read_csv(DATA_DIR / "final_combined_movies_5genres.csv")
        
        # Basic statistics
        total_samples = len(df)
        df['genre'] = df['genre'].str.split(',')
        genre_counts = df['genre'].explode().value_counts()
        
        # Text statistics
        df['synopsis_length'] = df['sinopsis'].str.len()
        avg_length = df['synopsis_length'].mean()
        max_length = df['synopsis_length'].max()
        min_length = df['synopsis_length'].min()
        
        # Multi-label statistics
        genre_combinations = df['genre'].apply(lambda x: ','.join(sorted(x)))
        common_combinations = genre_combinations.value_counts().head(5)
        
        return {
            'total_samples': total_samples,
            'genre_distribution': genre_counts.to_dict(),
            'text_stats': {
                'avg_length': avg_length,
                'max_length': max_length,
                'min_length': min_length
            },
            'common_combinations': common_combinations.to_dict(),
            'sample_data': df.head(5).to_dict('records')
        }
    except Exception as e:
        st.error(f"Error loading dataset: {str(e)}")
        return None


def plot_predictions(predictions):
    df = pd.DataFrame(predictions)
    fig = go.Figure(data=[
        go.Bar(
            x=df['genre'],
            y=df['probability'],
            marker_color='rgb(78, 205, 196)',
            hovertemplate='%{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Genre Predictions',
        xaxis_title='Genre',
        yaxis_title='Probability',
        yaxis_tickformat=',.0%',
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def plot_training_history(history_data):
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=history_data['epochs'], 
        y=history_data['training_loss'],
        name='Training Loss',
        line=dict(color='rgb(49, 130, 189)', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=history_data['epochs'],
        y=history_data['validation_loss'],
        name='Validation Loss',
        line=dict(color='rgb(189, 189, 189)', width=2, dash='dash')
    ))
    
    fig.add_trace(go.Scatter(
        x=history_data['epochs'],
        y=history_data['accuracy'],
        name='Accuracy',
        line=dict(color='rgb(44, 160, 44)', width=2),
        yaxis='y2'
    ))

    fig.update_layout(
        title='Training History',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        yaxis2=dict(
            title='Accuracy',
            overlaying='y',
            side='right',
            tickformat=',.0%',
            range=[0, 1]
        ),
        height=400,
        template='plotly_dark',
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig

def display_dataset_info():
    """Display dataset information and visualizations"""
    st.markdown("""
        <div class="card">
            <div class="card-title">Dataset Overview</div>
    """, unsafe_allow_html=True)
    
    dataset_info = load_dataset_info()
    if not dataset_info:
        return
    
    # Basic Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Samples", f"{dataset_info['total_samples']:,}")
    with col2:
        st.metric("Average Synopsis Length", f"{dataset_info['text_stats']['avg_length']:.0f} chars")
    with col3:
        st.metric("Number of Genres", f"{len(dataset_info['genre_distribution'])}")
    
    # Genre Distribution
    st.subheader("Genre Distribution")
    genre_df = pd.DataFrame(list(dataset_info['genre_distribution'].items()),
                           columns=['Genre', 'Count'])
    fig = px.bar(genre_df, x='Genre', y='Count',
                 title='Distribution of Genres',
                 template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Genre Combinations
    st.subheader("Common Genre Combinations")
    combinations_df = pd.DataFrame(list(dataset_info['common_combinations'].items()),
                                 columns=['Combination', 'Count'])
    fig = px.pie(combinations_df, values='Count', names='Combination',
                 title='Top 5 Genre Combinations',
                 template='plotly_dark')
    st.plotly_chart(fig, use_container_width=True)
    
    # Text Length Distribution
    st.subheader("Synopsis Length Statistics")
    stats_cols = st.columns(3)
    with stats_cols[0]:
        st.metric("Min Length", f"{dataset_info['text_stats']['min_length']:,} chars")
    with stats_cols[1]:
        st.metric("Average Length", f"{dataset_info['text_stats']['avg_length']:.0f} chars")
    with stats_cols[2]:
        st.metric("Max Length", f"{dataset_info['text_stats']['max_length']:,} chars")
    
    # Sample Data
    st.subheader("Sample Data")
    with st.expander("Show Sample Data"):
        for idx, sample in enumerate(dataset_info['sample_data'], 1):
            st.markdown(f"**Sample {idx}**")
            st.write(f"Synopsis: {sample['sinopsis'][:200]}...")
            st.write(f"Genres: {sample['genre']}")
            st.markdown("---")
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_system_info() -> Dict:
    """Get detailed system information including hardware and resource usage"""
    try:
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        system_info = {
            'cpu': {
                'cores_physical': psutil.cpu_count(logical=False),
                'cores_total': psutil.cpu_count(logical=True),
                'usage_percent': psutil.cpu_percent(),
                'frequency_current': cpu_freq.current if cpu_freq else None,
                'frequency_max': cpu_freq.max if cpu_freq else None
            },
            'memory': {
                'total': memory.total / (1024 ** 3),  # Convert to GB
                'available': memory.available / (1024 ** 3),
                'used': memory.used / (1024 ** 3),
                'percent': memory.percent
            },
            'disk': {
                'total': disk.total / (1024 ** 3),
                'used': disk.used / (1024 ** 3),
                'free': disk.free / (1024 ** 3),
                'percent': disk.percent
            },
            'gpu': None
        }
        
        if torch.cuda.is_available():
            system_info['gpu'] = {
                'name': torch.cuda.get_device_name(0),
                'count': torch.cuda.device_count(),
                'memory_allocated': torch.cuda.memory_allocated(0) / (1024 ** 2),  # MB
                'memory_cached': torch.cuda.memory_reserved(0) / (1024 ** 2),
                'max_memory': torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            }
        
        return system_info
    except Exception as e:
        logging.error(f"Error getting system info: {str(e)}")
        return {}

def display_resource_usage(system_info: Dict):
    """Display detailed resource usage information"""
    st.markdown("### 🖥️ Hardware Resources")
    
    # CPU Information
    cpu_info = system_info.get('cpu', {})
    st.markdown("#### CPU")
    cpu_cols = st.columns(4)
    with cpu_cols[0]:
        st.metric("Physical Cores", cpu_info.get('cores_physical', 'N/A'))
    with cpu_cols[1]:
        st.metric("Total Cores", cpu_info.get('cores_total', 'N/A'))
    with cpu_cols[2]:
        st.metric("CPU Usage", f"{cpu_info.get('usage_percent', 0)}%")
    with cpu_cols[3]:
        st.metric("CPU Frequency", 
                 f"{cpu_info.get('frequency_current', 0):.2f} MHz" if cpu_info.get('frequency_current') else 'N/A')
    
    # Memory Information
    memory_info = system_info.get('memory', {})
    st.markdown("#### Memory")
    memory_cols = st.columns(4)
    with memory_cols[0]:
        st.metric("Total RAM", f"{memory_info.get('total', 0):.1f} GB")
    with memory_cols[1]:
        st.metric("Used RAM", f"{memory_info.get('used', 0):.1f} GB")
    with memory_cols[2]:
        st.metric("Available RAM", f"{memory_info.get('available', 0):.1f} GB")
    with memory_cols[3]:
        st.metric("Memory Usage", f"{memory_info.get('percent', 0)}%")

    # Progress bar for memory usage
    st.progress(memory_info.get('percent', 0) / 100)
    
    # GPU Information if available
    gpu_info = system_info.get('gpu')
    if gpu_info:
        st.markdown("#### GPU")
        gpu_cols = st.columns(4)
        with gpu_cols[0]:
            st.metric("GPU Model", gpu_info.get('name', 'N/A'))
        with gpu_cols[1]:
            st.metric("GPU Memory Used", f"{gpu_info.get('memory_allocated', 0):.1f} MB")
        with gpu_cols[2]:
            st.metric("GPU Memory Cached", f"{gpu_info.get('memory_cached', 0):.1f} MB")
        with gpu_cols[3]:
            st.metric("Total GPU Memory", f"{gpu_info.get('max_memory', 0):.1f} MB")
        
        # Progress bar for GPU memory usage
        gpu_usage = gpu_info.get('memory_allocated', 0) / gpu_info.get('max_memory', 1) * 100
        st.progress(gpu_usage / 100)

def display_training_statistics(training_history: Dict):
    """Display detailed training statistics"""
    st.markdown("### 📊 Training Statistics")
    
    if 'training_stats' in training_history:
        stats = training_history['training_stats']
        
        # Training Duration
        if 'duration' in stats:
            duration = stats['duration']
            duration_cols = st.columns(3)
            with duration_cols[0]:
                st.metric("Total Training Time", 
                         f"{duration.get('total_seconds', 0)/3600:.2f} hours")
            with duration_cols[1]:
                st.metric("Average Epoch Time", 
                         f"{duration.get('average_epoch_time', 0)/60:.2f} minutes")
            with duration_cols[2]:
                total_epochs = len(training_history.get('training_history', {}).get('epochs', []))
                st.metric("Total Epochs", str(total_epochs))
        
        # Memory Usage Chart
        if 'memory_tracking' in stats:
            st.markdown("#### Memory Usage During Training")
            memory_df = pd.DataFrame(stats['memory_tracking'])
            fig = px.line(memory_df, 
                         x='timestamp', 
                         y='memory_usage',
                         title='Memory Usage Over Time',
                         template='plotly_dark')
            fig.update_layout(
                xaxis_title="Time",
                yaxis_title="Memory Usage (MB)"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Batch Processing Stats
        if 'batch_statistics' in stats:
            st.markdown("#### Batch Processing Performance")
            batch_df = pd.DataFrame(stats['batch_statistics'])
            
            stat_cols = st.columns(3)
            with stat_cols[0]:
                st.metric("Average Batch Time", 
                         f"{batch_df['time'].mean():.3f} seconds")
            with stat_cols[1]:
                st.metric("Max Batch Time", 
                         f"{batch_df['time'].max():.3f} seconds")
            with stat_cols[2]:
                st.metric("Min Batch Time", 
                         f"{batch_df['time'].min():.3f} seconds")
            
            # Batch processing time distribution
            fig = px.histogram(batch_df, 
                             x='time',
                             title='Batch Processing Time Distribution',
                             template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)

def display_model_info(training_history: Dict):
    """Display detailed model information"""
    st.markdown("### 🤖 Model Information")
    
    if 'model_info' in training_history:
        model_info = training_history['model_info']
        parameters = model_info.get('parameters', {})
        
        # Model Architecture
        st.markdown("#### Model Architecture")
        arch_cols = st.columns(3)
        with arch_cols[0]:
            st.metric("Total Parameters", 
                     f"{parameters.get('total_parameters', 0):,}")
        with arch_cols[1]:
            st.metric("Trainable Parameters", 
                     f"{parameters.get('trainable_parameters', 0):,}")
        with arch_cols[2]:
            st.metric("Number of Classes", 
                     str(len(model_info.get('classes', []))))
        
        # Model Configuration
        st.markdown("#### Training Configuration")
        config_cols = st.columns(4)
        with config_cols[0]:
            st.metric("Batch Size", Config.MODEL_PARAMS['BATCH_SIZE'])
        with config_cols[1]:
            st.metric("Learning Rate", f"{Config.MODEL_PARAMS['LEARNING_RATE']:.2e}")
        with config_cols[2]:
            st.metric("Weight Decay", Config.MODEL_PARAMS['WEIGHT_DECAY'])
        with config_cols[3]:
            st.metric("Max Length", Config.MODEL_PARAMS['MAX_LENGTH'])
        
        # Dataset Statistics
        if 'text_stats' in model_info:
            st.markdown("#### Dataset Statistics")
            text_stats = model_info['text_stats']
            stats_cols = st.columns(3)
            with stats_cols[0]:
                st.metric("Total Samples", 
                         f"{model_info.get('total_samples', 0):,}")
            with stats_cols[1]:
                st.metric("Average Token Length", 
                         f"{text_stats.get('avg_token_length', 0):.1f}")
            with stats_cols[2]:
                st.metric("Truncated Sequences", 
                         f"{text_stats.get('truncated_sequences', 0)} ({text_stats.get('truncation_percentage', 0):.1f}%)")

def display_gpu_monitoring():
    """Display real-time GPU monitoring information"""
    if torch.cuda.is_available():
        st.markdown("### 🎮 GPU Monitoring")
        
        # Get current GPU statistics
        current_memory = torch.cuda.memory_allocated(0) / (1024 ** 2)
        max_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        memory_percent = (current_memory / max_memory) * 100
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Current Memory Usage", f"{current_memory:.1f} MB",
                     delta=f"{memory_percent:.1f}% of total")
        with col2:
            st.metric("Memory Utilization", f"{memory_percent:.1f}%")
        
        # Memory usage gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=memory_percent,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "GPU Memory Utilization"},
            gauge={
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"},
                    {'range': [80, 100], 'color': "darkgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 80
                }
            }
        ))
        fig.update_layout(template='plotly_dark')
        st.plotly_chart(fig, use_container_width=True)

def display_detailed_training_info(training_history, evaluation_metrics, experiment_info):
    """Display detailed training information in collapsible sections"""
    st.markdown("""
        <div class="card">
            <div class="card-title">Detailed Training Information</div>
        """, unsafe_allow_html=True)
    
    # Training Process Details
    with st.expander("🔄 Training Process Details"):
        cols = st.columns(2)
        with cols[0]:
            st.markdown("### Augmentation Parameters")
            st.write("**Mixup Probability:**", Config.MODEL_PARAMS['MIXUP_PROB'])
            st.write("**Label Smoothing:**", Config.MODEL_PARAMS['SMOOTHING'])
            st.write("**Gradient Clipping:** 1.0")
            
        with cols[1]:
            st.markdown("### Optimization Settings")
            st.write("**Weight Decay:**", Config.MODEL_PARAMS['WEIGHT_DECAY'])
            st.write("**Early Stopping Patience:**", Config.MODEL_PARAMS['PATIENCE'])
            st.write("**Learning Rate Schedule:** ReduceLROnPlateau")
        
        # Memory Usage Plot
        if 'memory_usage' in training_history:
            st.markdown("### Memory Usage During Training")
            memory_df = pd.DataFrame(training_history['memory_usage'])
            st.line_chart(memory_df)
    
    # Optimization Details
    if 'hyperparameters' in experiment_info:
        with st.expander("🎯 Hyperparameter Optimization Details"):
            st.markdown("### Best Trial Parameters")
            for param, value in experiment_info['hyperparameters'].items():
                st.write(f"**{param}:** {value}")
            
            # Parameter importance plot if available
            if 'param_importances' in experiment_info:
                param_imp_df = pd.DataFrame(experiment_info['param_importances'].items(),
                                          columns=['Parameter', 'Importance'])
                fig = px.bar(param_imp_df, x='Parameter', y='Importance',
                           title='Hyperparameter Importance',
                           template='plotly_dark')
                st.plotly_chart(fig)
    
    # Per-Genre Analytics
    with st.expander("📊 Detailed Genre Analytics"):
        if 'per_genre' in evaluation_metrics:
            for genre, metrics in evaluation_metrics['per_genre'].items():
                st.markdown(f"### {genre}")
                cols = st.columns(4)
                cols[0].metric("Accuracy", f"{metrics['accuracy']:.3%}")
                cols[1].metric("F1 Score", f"{metrics['f1_score']:.3%}")
                cols[2].metric("Precision", f"{metrics['precision']:.3%}")
                cols[3].metric("Recall", f"{metrics['recall']:.3%}")
                
                # Add confusion matrix visualization
                if 'confusion_matrix' in metrics:
                    cm = metrics['confusion_matrix']
                    fig = px.imshow(cm, text=cm, aspect='auto',
                                  labels=dict(x="Predicted", y="Actual"),
                                  title=f"Confusion Metricsx - {genre}")
                    st.plotly_chart(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)

def display_optimization_results(experiment_info):
    """Display detailed optimization results"""
    st.markdown("""
        <div class="card">
            <div class="card-title">Optimization Results</div>
        """, unsafe_allow_html=True)
    
    if 'optimization_results' in experiment_info:
        opt_results = experiment_info['optimization_results']
        
        # Trial statistics
        cols = st.columns(3)
        cols[0].metric("Total Trials", opt_results.get('n_trials', 'N/A'))
        cols[1].metric("Completed Trials", opt_results.get('completed_trials', 'N/A'))
        cols[2].metric("Pruned Trials", opt_results.get('pruned_trials', 'N/A'))
        
        # Best trial details
        st.markdown("### Best Trial Details")
        if 'best_trial' in opt_results:
            best_trial = opt_results['best_trial']
            st.write(f"**Trial Number:** {best_trial.get('number', 'N/A')}")
            st.write(f"**Value:** {best_trial.get('value', 'N/A'):.4f}")
            
            # Parameters table
            if 'params' in best_trial:
                st.markdown("#### Parameters")
                param_df = pd.DataFrame([best_trial['params']])
                st.dataframe(param_df)
        
        # Optimization history plot
        if 'history' in opt_results:
            st.markdown("### Optimization History")
            history_df = pd.DataFrame(opt_results['history'])
            fig = px.line(history_df, x='trial', y='value',
                         title='Optimization Progress')
            st.plotly_chart(fig)
    
    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Movie Genre Classifier", page_icon="🎬", layout="wide")
    local_css()
    
    st.markdown("<h1>🎬 Movie Genre Classifier</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Multi-label Movie Genre Classification using IndoBERT</p>", unsafe_allow_html=True)
    
    try:
        experiments = get_all_experiments()
        experiment_names = [exp[0] for exp in experiments]
        selected_experiment = st.sidebar.selectbox(
            "Select Experiment",
            experiment_names,
            index=0
        )
        
        selected_exp_path = next(exp[1] for exp in experiments if exp[0] == selected_experiment)
        
        MODEL_PATH = selected_exp_path / 'model' / 'best_accuracy'
        TOKENIZER_PATH = selected_exp_path / 'tokenizer' / 'best_accuracy'
        METRICS_DIR = selected_exp_path / 'metrics'
        PLOTS_DIR = selected_exp_path / 'plots'
        CM_DIR = PLOTS_DIR / 'confusion_matrices'
        LOG_FILE = selected_exp_path / 'training.log'
        
        st.sidebar.info(f"Selected experiment from: {selected_experiment}")
        
        with st.spinner('Loading model and data...'):
            model, tokenizer, training_history, evaluation_metrics, experiment_info, device = load_model_and_history(
                MODEL_PATH, TOKENIZER_PATH, METRICS_DIR, selected_exp_path
            )
        
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
            "📊 Model Overview",
            "📈 Training Metrics",
            "📚 Training Log",
            "🔍 Evaluation Results",
            "🎯 Make Prediction",
            "📋 Dataset Information",
            "📝 Resource Information",  # Tab baru
            "⚙️ System Info"
        ])
        
        with tab1:
            st.markdown("""
                <div class="card">
                    <div class="card-title">Model Architecture</div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown("### Model Configuration")
                st.write("**Model Type:** IndoBERT")
                st.write("**Base Model:** indobenchmark/indobert-base-p1")
                st.write("**Problem Type:** Multi-label Classification")
                st.write("**Number of Classes:**", len(training_history['model_info']['classes']))
                st.write("**Max Sequence Length:**", Config.MODEL_PARAMS['MAX_LENGTH'])
                
            with cols[1]:
                st.markdown("### Training Parameters")
                st.write("**Batch Size:**", Config.OPTIM_PARAMS['BATCH_SIZE'])
                st.write("**Learning Rate:**", Config.OPTIM_PARAMS['LEARNING_RATE'])
                st.write("**Weight Decay:**", Config.OPTIM_PARAMS['WEIGHT_DECAY'])
                st.write("**Mixup Probability:**", Config.OPTIM_PARAMS['MIXUP_PROB'])
                st.write("**Label Smoothing:**", Config.OPTIM_PARAMS['SMOOTHING'])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display model performance summary
            st.markdown("""
                <div class="card">
                    <div class="card-title">Performance Summary</div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Best Accuracy", f"{training_history['training_history']['best_accuracy']:.3%}")
            with cols[1]:
                st.metric("Best Val Loss", f"{training_history['training_history']['best_val_loss']:.4f}")
            with cols[2]:
                st.metric("Total Epochs", len(training_history['training_history']['epochs']))
            with cols[3]:
                st.metric("Dataset Size", training_history['model_info']['total_samples'])
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display genre distribution
            st.markdown("""
                <div class="card">
                    <div class="card-title">Genre Distribution</div>
            """, unsafe_allow_html=True)
            
            genre_dist = pd.DataFrame(list(training_history['model_info']['genre_distribution'].items()),
                                    columns=['Genre', 'Count'])
            fig = px.bar(genre_dist, x='Genre', y='Count',
                        title='Distribution of Genres in Training Data',
                        template='plotly_dark')
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab2:
            st.markdown("""
                <div class="card">
                    <div class="card-title">Training Progress</div>
            """, unsafe_allow_html=True)
            
            # Training history plot
            st.plotly_chart(
                plot_training_history(training_history['training_history']),
                use_container_width=True
            )
            
            display_detailed_training_info(training_history, evaluation_metrics, experiment_info)
            
            cols = st.columns(2)
            with cols[0]:
                st.markdown("### Loss Analysis")
                loss_diff = np.array(training_history['training_history']['training_loss']) - \
                            np.array(training_history['training_history']['validation_loss'])
                
                # Menggunakan plotly untuk visualisasi yang lebih baik
                fig = px.line(
                    x=training_history['training_history']['epochs'],
                    y=loss_diff,
                    labels={'x': 'Epoch', 'y': 'Loss Difference'},
                    template='plotly_dark'
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)

            with cols[1]:
                st.markdown("### Training-Validation Loss")
                fig = go.Figure()
                
                fig.add_trace(go.Scatter(
                    x=training_history['training_history']['epochs'],
                    y=training_history['training_history']['training_loss'],
                    name='Training Loss',
                    line=dict(color='rgb(49, 130, 189)')
                ))
                
                fig.add_trace(go.Scatter(
                    x=training_history['training_history']['epochs'],
                    y=training_history['training_history']['validation_loss'],
                    name='Validation Loss',
                    line=dict(color='rgb(189, 189, 189)')
                ))
                
                fig.update_layout(
                    height=400,
                    template='plotly_dark',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("""
                <div class="card">
                    <div class="card-title">Training Log</div>
            """, unsafe_allow_html=True)
            
            try:
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    st.markdown('<div class="log-viewer">', unsafe_allow_html=True)
                    st.text(log_content)
                    st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading training log: {str(e)}")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab4:
            st.markdown("""
                <div class="card">
                    <div class="card-title">Model Evaluation</div>
            """, unsafe_allow_html=True)
            
            try:
                # Overall metrics
                st.markdown("### Overall Performance")
                cols = st.columns(4)
                
                # Safely get overall metrics with defaults
                overall = evaluation_metrics.get('overall', {})
                accuracy = overall.get('accuracy', 0)
                macro_f1 = overall.get('macro_f1', 0)
                macro_precision = overall.get('macro_precision', 0)
                macro_recall = overall.get('macro_recall', 0)
                
                cols[0].metric("Accuracy", f"{accuracy:.3%}")
                cols[1].metric("Macro F1", f"{macro_f1:.3%}")
                cols[2].metric("Macro Precision", f"{macro_precision:.3%}")
                cols[3].metric("Macro Recall", f"{macro_recall:.3%}")

                display_optimization_results(experiment_info)
                
                # Per-genre metrics
                st.markdown("### Per-Genre Performance")
                genre_metrics = evaluation_metrics.get('per_genre', {})
                metrics_data = []
                
                for genre, metrics in genre_metrics.items():
                    metrics_data.append({
                        'Genre': genre,
                        'Accuracy': f"{metrics.get('accuracy', 0):.3%}",
                        'F1 Score': f"{metrics.get('f1_score', 0):.3%}",
                        'Precision': f"{metrics.get('precision', 0):.3%}",
                        'Recall': f"{metrics.get('recall', 0):.3%}"
                    })
                
                if metrics_data:
                    df_metrics = pd.DataFrame(metrics_data)
                    st.dataframe(df_metrics, use_container_width=True)
                else:
                    st.warning("No per-genre metrics available.")
                
                # Display confusion matrices if available
                if CM_DIR.exists():
                    st.markdown("### Confusion Metrics")
                    cm_files = list(CM_DIR.glob('*.png'))
                    if cm_files:
                        cols = st.columns(2)
                        for idx, cm_file in enumerate(cm_files):
                            with cols[idx % 2]:
                                st.image(str(cm_file))
                    else:
                        st.info("No confusion metrics available.")
                
            except Exception as e:
                st.error(f"Error displaying evaluation metrics: {str(e)}")
                st.warning("Some metrics may be unavailable or in an incorrect format.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab5:
            st.markdown("""
                <div class="card">
                    <div class="card-title">Make Prediction</div>
            """, unsafe_allow_html=True)
            
            text_input = st.text_area(
                "Enter movie synopsis here...",
                height=200,
                placeholder="Type or paste your movie synopsis here..."
            )
            
            col1, col2 = st.columns([1, 5])
            with col1:
                predict_button = st.button("🎯 Predict")
            
            if predict_button and text_input:
                with st.spinner('Analyzing synopsis...'):
                    # Start time measurement
                    start_time = time.time()
                    
                    # Run prediction
                    predictions = predict_genre(text_input, model, tokenizer, device, METRICS_DIR)
                    
                    # End time measurement
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    
                    # Show elapsed time
                    st.info(f"⏱️ Prediction completed in {elapsed_time:.2f} seconds")
                    
                    # Show predictions visualization
                    st.plotly_chart(plot_predictions(predictions), use_container_width=True)
                    
                    # Confidence levels
                    high_conf = [p for p in predictions if p['probability'] >= 0.8]
                    med_conf = [p for p in predictions if 0.5 <= p['probability'] < 0.8]
                    low_conf = [p for p in predictions if p['probability'] < 0.5]
                    
                    # Display predictions by confidence level
                    if high_conf:
                        st.markdown("### High Confidence Predictions (≥ 80%)")
                        for pred in high_conf:
                            st.markdown(f"✨ **{pred['genre']}**: {pred['probability']:.1%}")
                    
                    if med_conf:
                        st.markdown("### Medium Confidence Predictions (50-80%)")
                        for pred in med_conf:
                            st.markdown(f"📍 **{pred['genre']}**: {pred['probability']:.1%}")
                    
                    if low_conf:
                        with st.expander("Show Low Confidence Predictions (< 50%)"):
                            for pred in low_conf:
                                st.markdown(f"❔ **{pred['genre']}**: {pred['probability']:.1%}")
            
            elif predict_button:
                st.error('Please enter a synopsis to make a prediction.')
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        with tab6:
            display_dataset_info()
            
            
        with tab7:
            display_system_info()
            
            st.markdown("""
                <div class="card">
                    <div class="card-title">Memory Usage</div>
            """, unsafe_allow_html=True)
            
            mem_usage = get_memory_usage()
            st.metric("Current Memory Usage", f"{mem_usage:.2f} MB")
            
            if torch.cuda.is_available():
                gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024 / 1024
                gpu_mem_cached = torch.cuda.memory_reserved(0) / 1024 / 1024
                st.metric("GPU Memory Allocated", f"{gpu_mem_allocated:.2f} MB")
                st.metric("GPU Memory Cached", f"{gpu_mem_cached:.2f} MB")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Additional system monitoring
            st.markdown("""
                <div class="card">
                    <div class="card-title">Resource Monitor</div>
            """, unsafe_allow_html=True)
            
            cols = st.columns(3)
            with cols[0]:
                cpu_percent = psutil.cpu_percent()
                st.metric("CPU Usage", f"{cpu_percent}%")
            
            with cols[1]:
                virtual_memory = psutil.virtual_memory()
                st.metric("RAM Usage", f"{virtual_memory.percent}%")
            
            with cols[2]:
                disk_usage = psutil.disk_usage('/')
                st.metric("Disk Usage", f"{disk_usage.percent}%")
            
            st.markdown("</div>", unsafe_allow_html=True)

        with tab8:
            st.markdown("""
                <div class="card">
                    <div class="card-title">System Information and Analysis</div>
                """, unsafe_allow_html=True)
            
            # Display basic system info
            display_resource_usage(display_system_info())
            
            st.markdown("</div>", unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required files and models are in the correct location.")
    
    finally:
        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()