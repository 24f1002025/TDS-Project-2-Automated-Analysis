# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas", "matplotlib", "seaborn", "httpx"
# ]
# ///

import os
import sys
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid GUI dependencies
import matplotlib.pyplot as plt
import seaborn as sns
import httpx
from typing import Dict, Any

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
        
        # Load data
        try:
            self.df = pd.read_csv(file_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")
        
        # Create output directory based on input file name
        self.output_dir = file_path.replace('.csv', '_analysis')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # AI Proxy token
        self.api_token = "##########################"

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
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'numeric_distributions.png'))
            plt.close()
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                plt.figure(figsize=(10, 8))
                correlation_matrix = self.df[numeric_cols].corr()
                sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
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
        headers = {"Authorization": f"Bearer {self.api_token}"}
        url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
        
        prompt = f"""
        Comprehensive Dataset Analysis Report:
        
        📊 Dataset Overview:
        - Total Rows: {analysis_details['total_rows']}
        - Total Columns: {analysis_details['total_columns']}
        
        🔍 Column Insights:
        Numeric Columns: {', '.join(analysis_details['numeric_columns'])}
        Categorical Columns: {', '.join(analysis_details['categorical_columns'])}
        
        📈 Key Statistics:
        {json.dumps(analysis_details['basic_stats'], indent=2)}
        
        🕵️ Data Quality Check:
        Missing Values:
        {json.dumps(analysis_details['missing_values'], indent=2)}
        
        Request:
        1. Provide a concise, engaging narrative about the dataset
        2. Highlight key observations and potential insights
        3. Suggest potential data science approaches or analyses
        4. Write in a markdown-friendly, professional format
        """
        
        data = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}]
        }
        
        try:
            response = httpx.post(url, json=data, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            return f"## LLM Analysis Failed\n\nError: {str(e)}\n\nPlease review the dataset manually."

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
