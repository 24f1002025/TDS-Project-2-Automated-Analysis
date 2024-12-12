#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas", "matplotlib", "seaborn", "httpx", "chardet", 
#   "scikit-learn", "statsmodels"
# ]
# ///

import os
import sys
import json
import base64
import chardet
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
from typing import Dict, Any, List, Optional
import numpy as np
import glob
from datetime import datetime
import logging

# Advanced analysis imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm

class DataAnalyzer:
    """
    A comprehensive data analysis tool that processes CSV files and generates insights.
    
    The class is organized into the following sections:
    1. Initialization and Setup
    2. Data Loading and Preprocessing
    3. Basic Analysis Methods
    4. Advanced Analysis Methods
    5. Visualization Methods
    6. Reporting Methods
    7. LLM Integration Methods
    8. Utility Methods

    Attributes:
        df (pd.DataFrame): The loaded dataset
        output_dir (str): Directory for analysis outputs
        api_token (str): Token for API authentication
        column_types (Dict): Mapping of columns to their data types
    """

    def __init__(self, file_path: str):
        """
        Initialize the data analyzer with a CSV file
        
        Args:
            file_path (str): Path to the CSV file
        
        Raises:
            FileNotFoundError: If the specified file doesn't exist
            ValueError: If the file is not a CSV or API token is invalid
        """
        # Validate input file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.endswith('.csv'):
            raise ValueError("Input must be a CSV file")
        
        # Load data with robust encoding detection
        self.df = self.read_csv_with_fallback(file_path)
        
        # Create output directory based on input file name
        self.output_dir = file_path.replace('.csv', '_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get API token securely
        self.api_token = self.get_api_token()
        
        # Additional attributes for story generation
        self.column_types = {col: str(dtype) for col, dtype in self.df.dtypes.items()}


    # All existing methods from the previous implementation remain the same:
    # - detect_encoding()
    # - read_csv_with_fallback()
    # - get_api_token()
    # - analyze_data_structure()
    # - generate_visualizations()
    # - generate_readme()
    def detect_encoding(self, file_path: str) -> str:
        """
        Detect file encoding using chardet
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            str: Detected encoding
        """
        with open(file_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
        
        # Print detected encoding for debugging
        print(f"🔍 Detected Encoding: {result['encoding']} (Confidence: {result['confidence']})")
        
        # Fallback to utf-8 if confidence is low
        return result['encoding'] if result['confidence'] > 0.8 else 'utf-8'

    def read_csv_with_fallback(self, file_path: str) -> pd.DataFrame:
        """
        Read CSV file with multiple encoding fallback strategies
        
        Args:
            file_path (str): Path to the CSV file
        
        Returns:
            pd.DataFrame: Loaded DataFrame
        """
        # List of encodings to try
        encodings = [
            'utf-8', 
            'latin-1', 
            'iso-8859-1', 
            'cp1252', 
            'utf-16', 
            'big5', 
            'shift_jis'
        ]

        # First, try auto-detection
        try:
            detected_encoding = self.detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=detected_encoding)
            return df
        except Exception as detection_error:
            print(f"❗ Encoding detection failed: {detection_error}")

        # Fallback to manual encoding attempts
        for encoding in encodings:
            try:
                print(f"Attempting to read file with {encoding} encoding...")
                df = pd.read_csv(file_path, encoding=encoding)
                print(f"✅ Successfully read file with {encoding} encoding")
                return df
            except Exception as e:
                print(f"❌ Failed with {encoding} encoding: {e}")
        
        # Ultimate fallback
        raise ValueError("Could not read CSV file with any known encoding")

    def get_api_token(self) -> str:
        """
        Retrieve API token from environment variable
        
        Returns:
            str: API token
        """
        # Retrieve token from environment variable
        api_token = os.environ.get('AIPROXY_TOKEN')
        
        # Validate token
        if not api_token:
            raise ValueError(
                "AIPROXY_TOKEN environment variable is not set. "
                "Please set it before running the script."
            )
        
        # Optional: Basic token format validation 
        if len(api_token) < 10:
            raise ValueError("Invalid API token format")
        
        return api_token

    def analyze_data_structure(self) -> Dict[str, Any]:
        """
        Perform comprehensive data structure analysis
        
        Returns:
            Dict containing data insights including:
            - total_rows: Number of rows in dataset
            - total_columns: Number of columns
            - column_types: Data type of each column
            - missing_values: Count of missing values per column
            - numeric_columns: List of numeric column names
            - categorical_columns: List of categorical column names
            - basic_stats: Basic statistics for numeric columns
        """
        # Basic statistics
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Convert dtypes to strings for JSON serialization
        column_types = {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        
        # Handle basic stats with safe conversion
        basic_stats = {}
        if len(numeric_cols) > 0:
            numeric_stats = self.df[numeric_cols].describe()
            basic_stats = {
                col: {stat: float(value) for stat, value in numeric_stats.loc[stat].items()}
                for col in numeric_stats.columns
                for stat in numeric_stats.index
            }
        
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "column_types": column_types,
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": list(numeric_cols),
            "categorical_columns": list(categorical_cols),
            "basic_stats": basic_stats
        }

    def generate_visualizations(self):
        """
        Create multiple visualizations with error handling
        
        Generates:
            - Distribution plots for numeric columns
            - Correlation heatmap for numeric columns
            - Missing value heatmap
            - Categorical value counts plots
        """
        plt.close('all')  # Ensure all previous plots are closed
        
        try:
            # Numeric columns visualization
            self._generate_numeric_distributions()
            
            # Correlation heatmap
            self._generate_correlation_heatmap()
            
            # Missing values heatmap
            self._generate_missing_values_heatmap()
            
            # Categorical plots
            self._generate_categorical_plots()
            
        except Exception as e:
            print(f"❌ Error generating visualizations: {str(e)}")

    def _generate_numeric_distributions(self):
        """Generate distribution plots for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            plt.figure(figsize=(15, 5 * ((len(numeric_cols) + 2) // 3)))
            for i, col in enumerate(numeric_cols, 1):
                plt.subplot((len(numeric_cols) + 2) // 3, 3, i)
                sns.histplot(self.df[col], kde=True)
                plt.title(f'Distribution of {col}')
                plt.xlabel(col)
                plt.ylabel('Frequency')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'numeric_distributions.png'))
            plt.close()

    def _generate_correlation_heatmap(self):
        """Generate correlation heatmap for numeric columns"""
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 1:
            plt.figure(figsize=(10, 8))
            correlation_matrix = self.df[numeric_cols].corr()
            sns.heatmap(correlation_matrix, 
                       annot=True, 
                       cmap='coolwarm', 
                       linewidths=0.5, 
                       center=0, 
                       vmin=-1, 
                       vmax=1)
            plt.title('Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
            plt.close()

    def _generate_missing_values_heatmap(self):
        """Generate missing values heatmap"""
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.df.isnull(), 
                   yticklabels=False, 
                   cmap='viridis', 
                   cbar_kws={'label': 'Missing Values'})
        plt.title('Missing Values Heatmap')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'missing_values_heatmap.png'))
        plt.close()

    def _generate_categorical_plots(self):
        """Generate plots for categorical columns"""
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if len(categorical_cols) > 0:
            plt.figure(figsize=(15, 5 * ((len(categorical_cols) + 2) // 3)))
            for i, col in enumerate(categorical_cols, 1):
                plt.subplot((len(categorical_cols) + 2) // 3, 3, i)
                value_counts = self.df[col].value_counts()
                if len(value_counts) > 10:
                    # Show only top 10 categories if there are too many
                    value_counts = value_counts.head(10)
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f'Value Counts for {col}')
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'categorical_plots.png'))
            plt.close()

    def save_plot(self, fig, filename: str):
        """
        Safely save a matplotlib figure
        
        Args:
            fig: matplotlib figure object
            filename (str): Name of the file to save
        """
        try:
            fig.savefig(os.path.join(self.output_dir, filename))
            plt.close(fig)
        except Exception as e:
            print(f"❌ Error saving plot {filename}: {str(e)}")

    def generate_readme(self, narrative: str):
        """
        Generate comprehensive README
        """
        readme_path = os.path.join(self.output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Dataset Analysis Report\n\n")
            f.write(narrative)


    # SECTION 3: Advanced Analysis Methods
    def perform_advanced_analysis(self, data_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform advanced statistical and machine learning analyses
        
        Args:
            data_insights (Dict): Insights from initial data analysis
        
        Returns:
            Dict containing advanced analysis results including:
            - outliers: Outlier detection results
            - clustering: K-means clustering results
            - pca: Principal Component Analysis results
            - time_series: Time series analysis (if applicable)
        """
        advanced_insights = {}
        numeric_cols = data_insights['numeric_columns']
        
        if len(numeric_cols) > 1:
            # Outlier Detection
            advanced_insights['outliers'] = self.detect_outliers(numeric_cols)
            
            # Clustering Analysis
            advanced_insights['clustering'] = self.perform_clustering(numeric_cols)
            
            # Dimensionality Reduction
            advanced_insights['pca'] = self.perform_pca(numeric_cols)
            
            # Time Series Analysis (if applicable)
            if self.is_time_series_data(numeric_cols):
                advanced_insights['time_series'] = self.perform_time_series_analysis()
        
        return advanced_insights

    def detect_outliers(self, columns: List[str]) -> Dict:
        """
        Detect outliers using IQR method
        
        Args:
            columns (List[str]): Numeric columns to analyze
        
        Returns:
            Dict containing outlier information for each column:
            - count: Number of outliers
            - percentage: Percentage of outliers
            - lower_bound: Lower threshold for outliers
            - upper_bound: Upper threshold for outliers
        """
        outliers = {}
        for col in columns:
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            column_outliers = self.df[(self.df[col] < lower_bound) | (self.df[col] > upper_bound)]
            outliers[col] = {
                'count': len(column_outliers),
                'percentage': len(column_outliers) / len(self.df) * 100,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
        return outliers

    def perform_clustering(self, columns: List[str], n_clusters: int = 3) -> Dict:
        """
        Perform K-means clustering with missing value handling
        
        Args:
            columns (List[str]): Columns to use for clustering
            n_clusters (int): Number of clusters to form
        
        Returns:
            Dict containing clustering results:
            - cluster_centers: Coordinates of cluster centers
            - inertia: Sum of squared distances to closest cluster center
        """
        # Prepare data
        X = self.df[columns].copy()
        X.fillna(X.mean(), inplace=True)
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)
        
        # Visualize clustering
        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
        plt.title('Clustering Analysis')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1] if len(columns) > 1 else columns[0])
        plt.savefig(os.path.join(self.output_dir, 'clustering_analysis.png'))
        plt.close()
        
        return {
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': float(kmeans.inertia_)
        }

    def perform_pca(self, columns: List[str]) -> Dict:
        """
        Perform Principal Component Analysis with missing value handling
        
        Args:
            columns (List[str]): Columns to use for PCA
        
        Returns:
            Dict containing PCA results:
            - explained_variance_ratio: Proportion of variance explained by each component
            - cumulative_variance: Cumulative proportion of variance explained
        """
        X = self.df[columns].copy()
        X.fillna(X.mean(), inplace=True)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA()
        pca_result = pca.fit_transform(X_scaled)
        
        # Visualize explained variance
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                pca.explained_variance_ratio_.cumsum(), 'bo-')
        plt.title('Cumulative Explained Variance')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.savefig(os.path.join(self.output_dir, 'pca_variance.png'))
        plt.close()
        
        return {
            'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
            'cumulative_variance': pca.explained_variance_ratio_.cumsum().tolist()
        }

    def is_time_series_data(self, columns: List[str]) -> bool:
        """
        Check if the dataset appears to be a time series
        
        Args:
            columns (List[str]): Columns to check
        
        Returns:
            bool: Whether the data appears to be time series
        """
        return any('date' in col.lower() or 'time' in col.lower() for col in columns)

    def perform_time_series_analysis(self) -> Dict:
        """
        Perform basic time series analysis
        
        Returns:
            Dict with time series insights
        """
        # Placeholder for time series analysis
        return {
            'analysis': 'Requires more sophisticated time series detection'
        }

    def encode_image(self, image_path: str) -> str:
        """
        Encode image to base64 for LLM vision analysis
        
        Args:
            image_path (str): Path to the image file
        
        Returns:
            str: Base64 encoded image
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image {image_path}: {str(e)}")
            return ""
    
    def _format_data_coverage(self, coverage: Dict[str, float]) -> str:
        """
        Format data coverage information
        
        Args:
            coverage (Dict): Data coverage dictionary with column names as keys
                           and coverage percentages as values
        
        Returns:
            str: Formatted coverage information in markdown bullet points
        """
        coverage_lines = [
            f"- **{col}**: {coverage*100:.2f}% covered" 
            for col, coverage in coverage.items()
        ]
        return "\n".join(coverage_lines)
    
    def _format_column_types(self, column_types: Dict[str, str]) -> str:
        """
        Format column types for README
        
        Args:
            column_types (Dict): Column type dictionary with column names as keys
                               and data types as values
        
        Returns:
            str: Formatted column types in markdown bullet points
        """
        type_lines = [
            f"- **{col}**: {dtype}" 
            for col, dtype in column_types.items()
        ]
        return "\n".join(type_lines)
    
    def _format_unique_values(self, unique_values: Dict[str, int]) -> str:
        """
        Format unique values information
        
        Args:
            unique_values (Dict): Dictionary containing count of unique values
                                for each column
        
        Returns:
            str: Formatted unique values in markdown bullet points
        """
        unique_lines = [
            f"- **{col}**: {count} unique values" 
            for col, count in unique_values.items()
        ]
        return "\n".join(unique_lines)

    def analyze_images_with_llm(self, image_paths: List[str]) -> str:
        """
        Analyze generated images using LLM vision
        
        Args:
            image_paths (List[str]): Paths to image files
        
        Returns:
            str: LLM's image analysis
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        
        # Prepare multi-modal content
        content = [
            {"type": "text", "text": "Analyze these visualizations and provide insights:"},
            *[
                {
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/png;base64,{self.encode_image(path)}", 
                        "detail": "low"
                    }
                } 
                for path in image_paths
            ]
        ]
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1500
        }
        
        try:
            response = httpx.post(url, json=data, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"Image analysis failed: {str(e)}"

    def query_llm_with_function_calling(self, analysis_details: Dict[str, Any]) -> str:
        """
        Enhanced LLM query with function calling capabilities
        
        Args:
            analysis_details (Dict): Analyzed data details
        
        Returns:
            str: LLM generated narrative or function call result
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "user", "content": json.dumps(analysis_details)},
                {"role": "system", "content": "Suggest appropriate analysis functions based on the dataset."}
            ],
            "functions": self.get_analysis_functions(),
            "function_call": "auto",
            "max_tokens": 1000
        }
        
        try:
            response = httpx.post(url, json=data, headers=headers, timeout=30.0)
            response.raise_for_status()
            
            # Extract function call and generate narrative
            response_data = response.json()
            function_call = response_data.get('choices', [{}])[0].get('function_call', {})
            
            if function_call:
                # Process function suggestion
                suggested_function = function_call.get('name', '')
                suggested_params = json.loads(function_call.get('arguments', '{}'))
                
                # Call the suggested function with parameters
                if hasattr(self, suggested_function):
                    return getattr(self, suggested_function)(**suggested_params)
            
            return response_data['choices'][0]['message']['content']
        except Exception as e:
            return f"Function calling failed: {str(e)}"

    def get_analysis_functions(self) -> List[Dict]:
        """
        Define available analysis functions for LLM
        
        Returns:
            List of function definitions with parameters
        """
        return [
            {
                "name": "perform_correlation_analysis",
                "description": "Analyze correlations between numeric columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "columns": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Numeric columns to analyze correlations"
                        }
                    }
                }
            },
            {
                "name": "detect_outliers",
                "description": "Identify outliers in numeric columns",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "method": {
                            "type": "string",
                            "enum": ["IQR", "Z-Score", "Modified Z-Score"],
                            "description": "Outlier detection method"
                        }
                    }
                }
            }
        ]

    def run_analysis(self):
        """
        Orchestrate the entire data analysis workflow with dynamic function calling
        
        Returns:
            Dict containing all analysis results and insights
        """
        try:
            # Step 1: Analyze data structure
            self._log_progress("Analyzing data structure...")
            data_insights = self.analyze_data_structure()
            
            # Step 2: Generate visualizations
            self._log_progress("Generating visualizations...")
            self.generate_visualizations()
            
            # Step 3: Perform advanced analysis
            self._log_progress("Performing advanced analysis...")
            advanced_insights = self.perform_advanced_analysis(data_insights)
            
            # Step 4: Generate narrative
            self._log_progress("Generating narrative...")
            narrative = self.generate_story()
            
            # Step 5: Analyze generated images
            self._log_progress("Analyzing visualizations...")
            image_paths = [
                os.path.join(self.output_dir, 'numeric_distributions.png'),
                os.path.join(self.output_dir, 'correlation_heatmap.png'),
                os.path.join(self.output_dir, 'clustering_analysis.png'),
                os.path.join(self.output_dir, 'pca_variance.png')
            ]
            image_analysis = self.analyze_images_with_llm(image_paths)
            
            # Step 6: Combine narrative with image analysis
            full_narrative = f"{narrative}\n\n## Visual Analysis\n{image_analysis}"
            
            # Step 7: Generate README
            self._log_progress("Generating README...")
            self.generate_readme(full_narrative)
            
            # Ensure all values are JSON serializable
            serializable_insights = self._make_json_serializable(advanced_insights)
            
            self._log_progress("Analysis completed successfully!")
            return {**data_insights, **serializable_insights}
            
        except Exception as e:
            self._log_progress(f"Error during analysis: {str(e)}", error=True)
            raise

    def generate_story(self) -> str:
        """
        Generate an adaptive, context-aware narrative
        
        Returns:
            str: Generated story narrative that describes the key insights 
                 and patterns found in the dataset
        """
        # Prepare story generation prompt
        story_prompt = self._prepare_story_prompt()
        
        # Query LLM for story generation
        story = self._query_llm_for_story(story_prompt)
        
        return story

    def _prepare_story_prompt(self) -> str:
        """
        Prepare a comprehensive prompt for story generation
        
        Returns:
            str: Detailed storytelling prompt incorporating dataset characteristics
        """
        # Analyze dataset characteristics
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Prepare interesting statistical insights
        insights = []
        for col in numeric_cols:
            mean = self.df[col].mean()
            median = self.df[col].median()
            std = self.df[col].std()
            insights.append(f"{col}: Mean={mean:.2f}, Median={median:.2f}, Std Dev={std:.2f}")
        
        # Categorical column insights
        cat_insights = []
        for col in categorical_cols:
            top_categories = self.df[col].value_counts().head(3)
            cat_insights.append(f"{col} Top Categories: {dict(top_categories)}")
        
        # Construct comprehensive prompt
        prompt = f"""
        Storytelling Challenge: Transform Dataset into a Compelling Narrative

        Dataset Overview:
        - Total Rows: {len(self.df)}
        - Total Columns: {len(self.df.columns)}
        - Column Types: {json.dumps(self.column_types, indent=2)}

        Numerical Insights:
        {chr(10).join(insights)}

        Categorical Insights:
        {chr(10).join(cat_insights)}

        Storytelling Objectives:
        1. Create an engaging narrative that reveals the dataset's hidden stories
        2. Use data points as narrative anchors
        3. Maintain scientific integrity while being creatively compelling
        4. Highlight unexpected patterns or interesting correlations
        5. Make the data come alive through storytelling

        Narrative Guidelines:
        - Begin with an intriguing overview
        - Use data as characters in the story
        - Explain complex insights in an accessible manner
        - Conclude with forward-looking implications
        """
        return prompt

    def _query_llm_for_story(self, prompt: str) -> str:
        """
        Query LLM for story generation
        
        Args:
            prompt (str): Storytelling prompt
        
        Returns:
            str: Generated story
        """
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system", 
                    "content": "You are a creative data storyteller. Transform data into an engaging narrative."
                },
                {
                    "role": "user", 
                    "content": prompt
                }
            ],
            "max_tokens": 1500,
            "temperature": 0.7
        }
        
        try:
            response = httpx.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions", 
                json=data, 
                headers=headers, 
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"""## 🚨 Story Generation Error

    Unable to generate story:
    - Error: {str(e)}

    **Fallback Narrative**: 
    This dataset holds untold stories waiting to be discovered. 
    While our AI storyteller encountered a challenge, 
    the data remains a treasure trove of insights."""

    def generate_readme(self, narrative: str):
        """
        Generate a comprehensive and engaging README
        
        Args:
            narrative (str): Generated narrative
        """
        readme_path = os.path.join(self.output_dir, 'README.md')
        
        # Determine dataset type and create a captivating title
        dataset_name = os.path.basename(self.output_dir).replace('_analysis', '')
        
        # Generate dataset profile for additional context
        dataset_profile = self._generate_dataset_profile()
        
        readme_content = f"""# 🔍 The Hidden Stories of {dataset_name.capitalize()} Data

    ## 📖 Data Journey: Unveiling Insights

    {narrative}

    ## 📊 Dataset Snapshot

    ### Overview
    - **Total Observations**: {dataset_profile['total_rows']} data points
    - **Exploratory Dimensions**: {dataset_profile['total_columns']} unique attributes

    ### Data Coverage
    {self._format_data_coverage(dataset_profile['data_coverage'])}

    ### Column Types
    {self._format_column_types(dataset_profile['column_types'])}

    ### Unique Values
    {self._format_unique_values(dataset_profile['unique_values'])}

    ### Missing Data
    {self._format_missing_data_summary(dataset_profile['missing_values'])}

    **Prepared with ❤️ by DataStory Explorer**
    """
        
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

    def _format_missing_data_summary(self, missing_values: Dict[str, int]) -> str:
        """
        Generate a summary of missing data
        
        Args:
            missing_values (Dict): Dictionary of missing values per column
        
        Returns:
            str: Formatted missing data summary
        """
        missing_summary = "| Column | Missing Values | Percentage |\n|--------|----------------|------------|\n"
        
        for col, missing in missing_values.items():
            percentage = (missing / len(self.df)) * 100
            missing_summary += f"| {col} | {missing} | {percentage:.2f}% |\n"
        
        return missing_summary

    def perform_correlation_analysis(self, columns: Optional[List[str]] = None) -> str:
        """
        Perform correlation analysis on specified columns
        
        Args:
            columns: Optional list of column names to analyze. If None, uses all numeric columns.
        
        Returns:
            str: Correlation analysis narrative
        """
        # If no columns specified, use all numeric columns
        if columns is None:
            columns = self.df.select_dtypes(include=['number']).columns
        
        # Compute correlation matrix
        correlation_matrix = self.df[columns].corr()
        
        # Prepare correlation narrative
        correlation_insights = []
        for i in range(len(columns)):
            for j in range(i+1, len(columns)):
                col1, col2 = columns[i], columns[j]
                correlation = correlation_matrix.loc[col1, col2]
                
                # Interpret correlation strength
                if abs(correlation) > 0.7:
                    strength = "strong"
                elif abs(correlation) > 0.4:
                    strength = "moderate"
                else:
                    strength = "weak"
                
                # Determine correlation direction
                direction = "positive" if correlation > 0 else "negative"
                
                correlation_insights.append(f"The correlation between {col1} and {col2} is {strength} ({direction}) with a value of {correlation:.2f}.")
        
        return "\n".join(correlation_insights)

    def _make_json_serializable(self, data: Dict) -> Dict:
        """
        Convert all values in a dictionary to JSON serializable format
        
        Args:
            data (Dict): Dictionary to convert
        
        Returns:
            Dict: Dictionary with all values converted to JSON serializable format
        """
        serializable = {}
        for key, value in data.items():
            try:
                json.dumps(value)  # Test if value is already serializable
                serializable[key] = value
            except (TypeError, OverflowError):
                if isinstance(value, (np.integer, np.floating)):
                    serializable[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable[key] = value.tolist()
                else:
                    serializable[key] = str(value)
        return serializable

    def _validate_output_path(self, path: str) -> bool:
        """
        Validate and ensure output path exists
        
        Args:
            path (str): Path to validate
        
        Returns:
            bool: True if path is valid and exists
        
        Raises:
            OSError: If path creation fails
        """
        try:
            os.makedirs(path, exist_ok=True)
            return os.path.exists(path)
        except Exception as e:
            print(f"❌ Error validating output path: {str(e)}")
            return False

    def _log_progress(self, message: str, error: bool = False):
        """
        Log progress messages with timestamps
        
        Args:
            message (str): Message to log
            error (bool): Whether this is an error message
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        prefix = "❌" if error else "ℹ️"
        print(f"{prefix} [{timestamp}] {message}")

    def _cleanup_resources(self):
        """
        Clean up any temporary resources and files
        
        This method:
        1. Closes all matplotlib figures
        2. Removes temporary files
        3. Cleans up any other resources
        """
        # Close all matplotlib figures
        plt.close('all')
        
        # Clean up any temporary files
        temp_files = glob.glob(os.path.join(self.output_dir, 'temp_*'))
        for file in temp_files:
            try:
                os.remove(file)
            except Exception as e:
                print(f"Warning: Could not remove temporary file {file}: {str(e)}")
        
        # Additional cleanup if needed
        try:
            # Clear any cached data
            if hasattr(self, '_cache'):
                self._cache.clear()
            
            # Close any open file handles
            for handler in logging.getLogger().handlers[:]:
                handler.close()
                
        except Exception as e:
            print(f"Warning: Error during cleanup: {str(e)}")

    def _generate_dataset_profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dataset profile
        
        Returns:
            Dict with dataset characteristics including:
            - total_rows: Number of rows
            - total_columns: Number of columns
            - column_types: Data types of each column
            - missing_values: Count of missing values
            - unique_values: Count of unique values
            - data_coverage: Percentage of non-null values
        """
        profile = {
            'total_rows': len(self.df),
            'total_columns': len(self.df.columns),
            'column_types': self.column_types,
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_values': {col: self.df[col].nunique() for col in self.df.columns},
            'data_coverage': {
                col: 1 - (self.df[col].isnull().sum() / len(self.df)) 
                for col in self.df.columns
            }
        }
        
        # Ensure all values are JSON serializable
        return self._make_json_serializable(profile)

def main():
    if len(sys.argv) < 2:
            print("Usage: python autolysis.py <path_to_csv>")
            sys.exit(1)
    
    try:
        analyzer = DataAnalyzer(sys.argv[1])
        results = analyzer.run_analysis()
        print(f"✅ Analysis complete. Check {sys.argv[1].replace('.csv', '_analysis')} directory.")
        print("\n📊 Quick Dataset Summary:")
        print(f"Total Rows: {results['total_rows']}")
        print(f"Total Columns: {results['total_columns']}")
        print(f"Numeric Columns: {', '.join(results.get('numeric_columns', []))}")
        print(f"Categorical Columns: {', '.join(results.get('categorical_columns', []))}")
        
        # Safely print advanced insights
        advanced_insights = results.get('advanced_insights', {})
        if advanced_insights:
            print("\n🔬 Advanced Insights:")
            for key, value in advanced_insights.items():
                print(f"{key}: {value}")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
[]
if __name__ == "__main__":
    main()
