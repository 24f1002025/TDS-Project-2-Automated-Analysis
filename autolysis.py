#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas", "matplotlib", "seaborn", "httpx", "chardet",
#   "scikit-learn", "statsmodels", "joblib"
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
import shutil
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import joblib

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
        Retrieve API token with fallback mechanism

        Returns:
            str: API token or fallback indicator
        """
        # Try environment variable first
        api_token = os.environ.get('AIPROXY_TOKEN')

        # If no token, use a fallback mechanism
        if not api_token:
            print("⚠️ AIPROXY_TOKEN not found. Falling back to limited analysis mode.")
            return "FALLBACK_MODE"

        # Validate token format if present
        if len(api_token) < 10:
            print("⚠️ Invalid API token. Falling back to limited analysis mode.")
            return "FALLBACK_MODE"

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
        Generate a comprehensive README with fallback mechanisms

        Args:
            narrative (str): Primary narrative generated by LLM or fallback method
        """
        readme_path = os.path.join(self.output_dir, 'README.md')

        # Prepare dataset profile
        dataset_profile = self.analyze_data_structure()

        # Construct README content with multiple sections
        readme_content = f"""# 🔍 Dataset Analysis Report: {os.path.basename(self.output_dir)}

## 📖 Executive Summary

{narrative}

## 🌐 Dataset Overview

### Basic Metrics
- **Total Observations**: {dataset_profile['total_rows']} records
- **Exploratory Dimensions**: {dataset_profile['total_columns']} attributes
- **Missing Values**: {sum(dataset_profile['missing_values'].values())} total

### Missing Data by Column

| Column | Missing Count |
|--------|---------------|
"""
        for col, missing in dataset_profile['missing_values'].items():
            readme_content += f"| {str(col)} | {str(missing)} |\n"

        readme_content += f"""

## 📊 Visualizations

Check the output directory for the following visualizations:
- Numeric distributions for numeric columns

## 🚨 Data Quality Insights

- Total missing values: {sum(dataset_profile['missing_values'].values())}
- Columns with the highest missing values: {max(dataset_profile['missing_values'], key=dataset_profile['missing_values'].get, default='None')} ({max(dataset_profile['missing_values'].values(), default=0)})

**Generated on**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Analysis Tool**: Automated Data Storyteller
"""

        # Write README
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

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
        Perform K-means clustering with missing value handling

        Args:
            columns (List[str]): Columns to use for clustering
            n_clusters (int): Number of clusters

        Returns:
            Dict with clustering results
        """
        # Prepare data
        X = self.df[columns].copy()

        # Handle missing values
        # Option 1: Simple imputation (replace NaNs with mean)
        X.fillna(X.mean(), inplace=True)

        # Alternative Option 2: Drop rows with NaN values
        # X.dropna(subset=columns, inplace=True)

        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(X_scaled)

        # Visualize clustering (use first two columns)
        plt.figure(figsize=(10, 6))
        plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=cluster_labels, cmap='viridis')
        plt.title('Clustering Analysis')
        plt.xlabel(columns[0])
        plt.ylabel(columns[1] if len(columns) > 1 else columns[0])
        plt.savefig(os.path.join(self.output_dir, 'clustering_analysis.png'))
        plt.close()

        return {
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'inertia': kmeans.inertia_
        }

    def perform_pca(self, columns: List[str]) -> Dict:
        """
        Perform Principal Component Analysis with missing value handling

        Args:
            columns (List[str]): Columns to use for PCA

        Returns:
            Dict with PCA results
        """
        X = self.df[columns].copy()

        # Handle missing values
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

    def analyze_images_with_llm(self, image_paths: List[str]) -> str:
        """
        Analyze generated images using LLM vision with comprehensive error handling

        Args:
            image_paths (List[str]): Paths to image files

        Returns:
            str: LLM's image analysis or fallback text
        """
        # Check if in fallback mode
        if self.api_token == "FALLBACK_MODE":
            return "🚫 Image analysis not available in fallback mode."

        # Validate image paths
        valid_image_paths = [path for path in image_paths if os.path.exists(path)]

        if not valid_image_paths:
            print("⚠️ No valid image paths found for analysis.")
            return "No images available for LLM analysis."

        try:
            # Prepare headers for API request
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json"
            }

            # API endpoint
            url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"

            # Prepare multi-modal content with comprehensive analysis prompt
            content = [
                {
                    "type": "text",
                    "text": """Perform a comprehensive analysis of these data visualization images:

                    Analysis Guidelines:
                    1. Describe the type and purpose of each visualization
                    2. Highlight key statistical insights
                    3. Identify any notable patterns or trends
                    4. Provide context and potential implications
                    5. Use a clear, concise, and engaging narrative style

                    Visualization Insights:"""
                },
                *[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{self.encode_image(path)}",
                            "detail": "high"  # Use high detail for more comprehensive analysis
                        }
                    }
                    for path in valid_image_paths
                ]
            ]

            # Prepare request data with enhanced parameters
            data = {
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 2000,  # Increased token limit for detailed analysis
                "temperature": 0.7,  # Balanced creativity
                "top_p": 0.9,        # Diverse response sampling
                "frequency_penalty": 0.5,  # Reduce repetition
                "presence_penalty": 0.5    # Encourage novel expressions
            }

            # Execute LLM request with enhanced error handling
            try:
                response = httpx.post(
                    url,
                    json=data,
                    headers=headers,
                    timeout=45.0  # Extended timeout for image analysis
                )

                # Raise an exception for bad responses
                response.raise_for_status()

                # Extract and process LLM response
                response_json = response.json()

                # Log token usage
                token_usage = response_json.get('usage', {})
                print(f"🔢 Image Analysis Token Usage: {token_usage}")

                # Extract image analysis
                image_analysis = response_json['choices'][0]['message']['content']

                # Validate analysis length
                if len(image_analysis.split()) < 50:  # Minimum word count
                    return self._generate_fallback_image_analysis(valid_image_paths)

                return image_analysis

            except httpx.RequestError as req_error:
                # Network-related errors
                print(f"🌐 Network Error during image analysis: {req_error}")
                return self._generate_fallback_image_analysis(valid_image_paths)

            except httpx.HTTPStatusError as http_error:
                # HTTP status errors
                print(f"🚨 HTTP Error during image analysis: {http_error}")
                return self._generate_fallback_image_analysis(valid_image_paths)

        except Exception as e:
            # Catch-all for unexpected errors
            print(f"❌ Unexpected Image Analysis Error: {e}")
            return self._generate_fallback_image_analysis(valid_image_paths)

    def _generate_fallback_image_analysis(self, image_paths: List[str]) -> str:
        """
        Generate a fallback image analysis based on file names and basic image properties

        Args:
            image_paths (List[str]): Paths to image files

        Returns:
            str: Fallback image analysis
        """
        fallback_analysis = "## 📊 Visualization Insights (Automated Analysis)\n\n"

        for path in image_paths:
            filename = os.path.basename(path)

            # Get image dimensions
            try:
                with PIL.Image.open(path) as img:
                    width, height = img.size
                    file_size = os.path.getsize(path)

                    fallback_analysis += f"### {filename}\n"
                    fallback_analysis += f"- Image Dimensions: {width}x{height} pixels\n"
                    fallback_analysis += f"- File Size: {file_size / 1024:.2f} KB\n"

                    # Basic interpretation based on filename
                    if 'distribution' in filename.lower():
                        fallback_analysis += "- Type: Distribution Plot\n"
                        fallback_analysis += "  * Represents the spread and frequency of data values\n"

                    elif 'correlation' in filename.lower():
                        fallback_analysis += "- Type: Correlation Heatmap\n"
                        fallback_analysis += "  * Shows relationships between different variables\n"

                    elif 'clustering' in filename.lower():
                        fallback_analysis += "- Type: Clustering Visualization\n"
                        fallback_analysis += "  * Illustrates data point groupings or patterns\n"

                    elif 'pca' in filename.lower():
                        fallback_analysis += "- Type: Principal Component Analysis (PCA) Plot\n"
                        fallback_analysis += "  * Demonstrates data dimensionality reduction\n"

            except Exception as e:
                fallback_analysis += f"- Unable to analyze {filename}: {str(e)}\n"

        return fallback_analysis

    def generate_story(self) -> str:
        """
        Generate an adaptive, context-aware narrative

        Returns:
            str: Generated story narrative
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
            str: Detailed storytelling prompt
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
        Query LLM for story generation with fallback

        Args:
            prompt (str): Storytelling prompt

        Returns:
            str: Generated story or fallback narrative
        """
        # Check if in fallback mode
        if self.api_token == "FALLBACK_MODE":
            return self._generate_fallback_narrative()

        try:
            # Existing LLM query logic
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

            response = httpx.post(
                "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
                json=data,
                headers=headers,
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']

        except Exception as e:
            print(f"⚠️ LLM Story Generation Failed: {e}")
            return self._generate_fallback_narrative()

    def run_analysis(self):
        """
        Orchestrate the entire data analysis workflow with dynamic function calling
        """
        try:
            print("🔍 Starting Data Analysis...")
            data_insights = self.analyze_data_structure()
            print("✅ Data Structure Analyzed")

            self.generate_visualizations()
            print("🖼️ Visualizations Generated")

            advanced_insights = self.perform_advanced_analysis(data_insights)
            print("🔬 Advanced Analysis Completed")

            narrative = self.generate_story() or self._generate_fallback_narrative()
            print("📖 Narrative Generated")

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

        except Exception as e:
            print(f"❌ Comprehensive Analysis Failed: {e}")
            import traceback
            traceback.print_exc()

    def _generate_fallback_narrative(self) -> str:
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        fallback_narrative = f"## 📊 Dataset Insights (Fallback Narrative)\n"

        for col in numeric_cols:
            fallback_narrative += f"- {col}: Mean={self.df[col].mean():.2f}, Median={self.df[col].median():.2f}, Std Dev={self.df[col].std():.2f}\n"

        return fallback_narrative

def main():
    if len(sys.argv) < 2:
        print("Usage: python autolysis.py <path_to_csv_or_directory>")
        sys.exit(1)

    input_path = sys.argv[1]

    # Determine if input is a directory or a single file
    if os.path.isdir(input_path):
        # If directory, find all CSV files
        csv_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if f.endswith('.csv')
        ]
    else:
        # If single file, use it directly
        csv_files = [input_path]

    # Track overall analysis results
    overall_results = []

    # Process each CSV file
    for csv_file in csv_files:
        try:
            print(f"\n🔍 Analyzing: {csv_file}")

            # Create DataAnalyzer instance
            analyzer = DataAnalyzer(csv_file)

            # Run analysis
            results = analyzer.run_analysis()

            # Print individual file analysis summary
            print(f"✅ Analysis complete. Check {csv_file.replace('.csv', '_analysis')} directory.")
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

            # Store results for potential further processing
            overall_results.append({
                'file': csv_file,
                'results': results
            })

            # Move README.md and PNG files to the eval directory
            eval_folder = os.path.join(os.path.dirname(csv_file), 'eval')
            os.makedirs(eval_folder, exist_ok=True)

            # Create a subdirectory based on the file name
            eval_subdir = os.path.join(eval_folder, os.path.basename(csv_file).replace('.csv', ''))
            os.makedirs(eval_subdir, exist_ok=True)

            # Move README.md
            readme_src = os.path.join(analyzer.output_dir, 'README.md')
            readme_dest = os.path.join(eval_subdir, 'README.md')
            shutil.move(readme_src, readme_dest)

            # Move PNG files
            png_files = [
                os.path.join(analyzer.output_dir, 'numeric_distributions.png'),
                os.path.join(analyzer.output_dir, 'correlation_heatmap.png'),
                os.path.join(analyzer.output_dir, 'clustering_analysis.png'),
                os.path.join(analyzer.output_dir, 'pca_variance.png')
            ]
            for png_file in png_files:
                if os.path.exists(png_file):
                    shutil.move(png_file, os.path.join(eval_subdir, os.path.basename(png_file)))

        except Exception as e:
            print(f"❌ Error analyzing {csv_file}: {e}")
            import traceback
            traceback.print_exc()

            # Optional: Continue processing other files even if one fails
            continue

    # Final summary if multiple files were processed
    if len(csv_files) > 1:
        print("\n🏁 Overall Analysis Summary:")
        print(f"Total files processed: {len(csv_files)}")
        print(f"Successfully analyzed: {len(overall_results)}")

    # Optional: Return overall results or exit
    if not overall_results:
        print("❌ No files were successfully analyzed.")
        sys.exit(1)

    return overall_results

if __name__ == "__main__":
    main()
