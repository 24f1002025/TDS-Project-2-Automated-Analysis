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

# Advanced analysis imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import statsmodels.api as sm

class DataAnalyzer:
    def __init__(self, file_path: str):
        """
        Initialize the data analyzer with a CSV file
        
        Args:
            file_path (str): Path to the CSV file
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
            Dict containing data insights
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
        """
        plt.close('all')  # Ensure all previous plots are closed
        
        # Numeric columns visualization
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        
        if len(numeric_cols) > 0:
            # Distribution plots
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
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = self.df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', 
                            linewidths=0.5, center=0, vmin=-1, vmax=1)
                plt.title('Correlation Matrix')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'correlation_heatmap.png'))
                plt.close()

    def generate_readme(self, narrative: str):
        """
        Generate comprehensive README
        """
        readme_path = os.path.join(self.output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Dataset Analysis Report\n\n")
            f.write(narrative)


    def perform_advanced_analysis(self, data_insights: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform advanced statistical and machine learning analyses
        
        Args:
            data_insights (Dict): Insights from initial data analysis
        
        Returns:
            Dict containing advanced analysis results
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
            Dict of outlier information
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
        Perform K-means clustering
        
        Args:
            columns (List[str]): Columns to use for clustering
            n_clusters (int): Number of clusters
        
        Returns:
            Dict with clustering results
        """
        # Prepare data
        X = self.df[columns]
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
        plt.ylabel(columns[1])
        plt.savefig(os.path.join(self.output_dir, 'clustering_analysis.png'))
        plt.close()
        
        return {
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_
        }

    def perform_pca(self, columns: List[str]) -> Dict:
        """
        Perform Principal Component Analysis
        
        Args:
            columns (List[str]): Columns to use for PCA
        
        Returns:
            Dict with PCA results
        """
        X = self.df[columns]
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
        # Simple heuristics to detect time series
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
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def analyze_images_with_llm(self, image_paths: List[ str]) -> str:
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
        Enhanced LLM query with function calling
        
        Args:
            analysis_details (Dict): Analyzed data details
        
        Returns:
            str: LLM generated narrative
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
                if suggested_function == "perform_correlation_analysis":
                    return self.perform_correlation_analysis(**suggested_params)
                elif suggested_function == "detect_outliers":
                    return self.detect_outliers(**suggested_params)
            
            return response_data['choices'][0]['message']['content']
        except Exception as e:
            return f"Function calling failed: {str(e)}"

    def get_analysis_functions(self) -> List[Dict]:
        """
        Dynamically generate function suggestions based on dataset
        
        Returns:
            List of function suggestions
        """
        functions = [
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
        return functions

    def run_analysis(self):
        """
        Orchestrate the entire data analysis workflow with dynamic function calling
        """
        # Analyze data structure
        data_insights = self.analyze_data_structure()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Perform advanced analysis
        advanced_insights = self.perform_advanced_analysis(data_insights)
        
        # Query LLM for narrative with function calling
        narrative = self.query_llm_with_function_calling(data_insights)
        
        # Get image paths for analysis
        image_paths = [
            os.path.join(self.output_dir, 'numeric_distributions.png'),
            os.path.join(self.output_dir, 'correlation_heatmap.png'),
            os.path.join(self.output_dir, 'clustering_analysis.png'),
            os.path.join(self.output_dir, 'pca_variance.png')
        ]
        
        # Analyze images with LLM
        image_analysis = self.analyze_images_with_llm(image_paths)
        
        # Combine narrative with image analysis
        full_narrative = narrative + "\n\n## Image Analysis\n" + image_analysis
        
        # Generate README
        self.generate_readme(full_narrative)
        
        # Ensure all values are JSON serializable
        serializable_insights = {}
        for key, value in advanced_insights.items():
            try:
                json.dumps(value)
                serializable_insights[key] = value
            except TypeError:
                serializable_insights[key] = str(value)
        
        return {**data_insights, **serializable_insights}

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

if __name__ == "__main__":
    main()
