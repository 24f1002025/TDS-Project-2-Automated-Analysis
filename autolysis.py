#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas", "matplotlib", "seaborn", "httpx", "chardet"
# ]
# ///

import os
import sys
import json
import chardet
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
from typing import Dict, Any, Optional

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
        
        return {
            "total_rows": len(self.df),
            "total_columns": len(self.df.columns),
            "column_types": self.df.dtypes.to_dict(),
            "missing_values": self.df.isnull().sum().to_dict(),
            "numeric_columns": list(numeric_cols),
            "categorical_columns": list(categorical_cols),
            "basic_stats": self.df[numeric_cols].describe().to_dict() if len(numeric_cols) > 0 else {}
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

    def query_llm(self, analysis_details: Dict[str, Any]) -> str:
        """
        Query LLM for data analysis narrative with enhanced error handling
        
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
        
        # Prepare a concise, informative prompt
        prompt = f"""
        Comprehensive Dataset Analysis Report:
        
        📊 Dataset Overview:
        - Total Rows: {analysis_details['total_rows']}
        - Total Columns: {analysis_details['total_columns']}
        
        🔍 Column Insights:
        Numeric Columns: {', '.join(analysis_details['numeric_columns'])}
        Categorical Columns: {', '.join(analysis_details['categorical_columns'])}
        
        📈 Key Statistics Summary:
        {json.dumps(analysis_details['basic_stats'], indent=2)}
        
        🕵️ Data Quality Snapshot:
        Missing Values:
        {json.dumps (analysis_details['missing_values'], indent=2)}
        
        Analysis Request:
        1. Craft an engaging, insightful narrative
        2. Highlight key data observations
        3. Suggest potential data science approaches
        4. Use professional, markdown-friendly format
        5. Focus on actionable insights
        """
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000
        }
        
        try:
            response = httpx.post(url, json=data, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except httpx.RequestError as e:
            return f"""## 🚨 LLM Communication Error

Unable to retrieve analysis due to network issues:
- Error: {str(e)}
- Please check your internet connection
- Verify AIPROXY_TOKEN is correct

**Recommendation**: 
1. Check network connectivity
2. Validate API token
3. Retry the analysis
"""
        except Exception as e:
            return f"""## 🛑 LLM Analysis Failed

An unexpected error occurred:
- Error: {str(e)}

**Manual Review Suggested**"""

    def generate_readme(self, narrative: str):
        """
        Generate comprehensive README
        """
        readme_path = os.path.join(self.output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("# Dataset Analysis Report\n\n")
            f.write(narrative)

    def run_analysis(self):
        """
        Orchestrate the entire data analysis workflow
        """
        # Analyze data structure
        data_insights = self.analyze_data_structure()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Query LLM for narrative
        narrative = self.query_llm(data_insights)
        
        # Generate README
        self.generate_readme(narrative)
        
        return data_insights

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
        print(f"Numeric Columns: {', '.join(results['numeric_columns'])}")
        print(f"Categorical Columns: {', '.join(results['categorical_columns'])}")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
