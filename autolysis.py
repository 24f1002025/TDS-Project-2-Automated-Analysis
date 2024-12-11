
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
from typing import Dict, Any, Optional, List
import numpy as np

class DataStoryteller:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize storyteller with adaptive capabilities
        
        Args:
            df (pd.DataFrame): Input dataset
        """
        self.df = df
        self.column_types = self._categorize_columns()
        self.dataset_profile = self._generate_dataset_profile()
    
    def _categorize_columns(self) -> Dict[str, str]:
        """
        Dynamically categorize columns based on data characteristics
        
        Returns:
            Dict mapping column names to inferred types
        """
        column_types = {}
        for col in self.df.columns:
            # Numeric type detection
            if pd.api.types.is_numeric_dtype(self.df[col]):
                if self.df[col].min() >= 0 and self.df[col].max() <= 1:
                    column_types[col] = 'percentage'
                elif any(keyword in col.lower() for keyword in ['age', 'year', 'date']):
                    column_types[col] = 'temporal'
                else:
                    column_types[col] = 'numeric'
            
            # Categorical type detection
            elif pd.api.types.is_categorical_dtype(self.df[col]) or \
                 self.df[col].dtype == 'object':
                # Check unique value count for categorization
                unique_count = self.df[col].nunique()
                if unique_count <= 10:
                    column_types[col] = 'categorical_low'
                elif unique_count <= 50:
                    column_types[col] = 'categorical_medium'
                else:
                    column_types[col] = 'categorical_high'
            
            # Date/Time type detection
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                column_types[col] = 'datetime'
        
        return column_types
    
    def _generate_dataset_profile(self) -> Dict[str, Any]:
        """
        Generate a comprehensive dataset profile
        
        Returns:
            Dict with dataset characteristics
        """
        return {
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
    
    def generate_story(self, api_token: str) -> str:
        """
        Generate an adaptive, context-aware narrative
        
        Args:
            api_token (str): API token for LLM query
        
        Returns:
            str: Generated story narrative
        """
        # Prepare story generation prompt
        story_prompt = self._prepare_story_prompt()
        
        # Query LLM for story generation
        story = self._query_llm_for_story(story_prompt, api_token)
        
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
    
    def _query_llm_for_story(self, prompt: str, api_token: str) -> str:
        """
        Query LLM for story generation
        
        Args:
            prompt (str): Storytelling prompt
            api_token (str): API token for authentication
        
        Returns:
            str: Generated story
        """
        headers = {
            "Authorization": f"Bearer {api_token}",
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
        Create multiple advanced visualizations with comprehensive labeling
        """
        plt.close('all')  # Ensure all previous plots are closed
        
        
        
        # Numeric columns visualization
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            # 1. Distribution Plots with Detailed Labeling
            plt.figure(figsize=(16, 6 * ((len(numeric_cols) + 2) // 3)), 
                    constrained_layout=True)
            for i, col in enumerate(numeric_cols, 1):
                ax = plt.subplot((len(numeric_cols) + 2) // 3, 3, i)
                
                # Calculate key statistics for annotation
                mean = self.df[col].mean()
                median = self.df[col].median()
                std = self.df[col].std()
                
                # Distribution plot with KDE
                sns.histplot(
                    self.df[col], 
                    kde=True, 
                    color='dodgerblue', 
                    alpha=0.6,
                    label='Distribution'
                )
                
                # Add vertical lines for mean and median
                plt.axvline(mean, color='red', linestyle='--', 
                            label=f'Mean: {mean:.2f}')
                plt.axvline(median, color='green', linestyle=':', 
                            label=f'Median: {median:.2f}')
                
                # Comprehensive title and labels
                plt.title(f'{col} Distribution', fontweight='bold')
                plt.xlabel(f'{col} Value', fontsize=10)
                plt.ylabel('Frequency', fontsize=10)
                
                # Add text box with statistics
                stats_text = (
                    f"μ (Mean): {mean:.2f}\n"
                    f"σ (Std Dev): {std:.2f}\n"
                    f"Median: {median:.2f}"
                )
                plt.text(0.95, 0.95, stats_text, 
                        transform=ax.transAxes, 
                        verticalalignment='top', 
                        horizontalalignment='right',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Add legend
                plt.legend(loc='best', title='Statistical Markers')
            
            plt.suptitle('Numeric Column Distributions', fontsize=16, fontweight='bold')
            plt.savefig(os.path.join(self.output_dir, 'detailed_distributions.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Correlation Heatmap with Comprehensive Labeling
            if len(numeric_cols) > 1:
                plt.figure(figsize=(14, 12))
                correlation_matrix = self.df[numeric_cols].corr()
                
                # Create heatmap with enhanced labeling
                sns.heatmap(
                    correlation_matrix, 
                    annot=True,  # Show correlation values
                    cmap='coolwarm',
                    center=0,
                    vmin=-1, 
                    vmax=1,
                    square=True,
                    linewidths=0.5,
                    cbar_kws={
                        "shrink": .8, 
                        "label": "Pearson Correlation Coefficient",
                        "orientation": "vertical"
                    },
                    annot_kws={
                        "fontsize": 8,
                        "fontweight": "bold"
                    }
                )
                
                plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
                plt.xlabel('Features', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'correlation_matrix.png'), 
                            dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Box Plot with Detailed Labeling
            plt.figure(figsize=(16, 8))
            sns.boxplot(
                data=self.df[numeric_cols], 
                palette='Set2'
            )
            plt.title('Boxplot of Numeric Columns', fontsize=16, fontweight='bold')
            plt.xlabel('Numeric Columns', fontsize=12)
            plt.ylabel('Value Distribution', fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'boxplot_comparison.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()
        
        # Categorical Column Visualizations
        if len(categorical_cols) > 0:
            # 4. Categorical Distribution with Comprehensive Labeling
            plt.figure(figsize=(16, 6 * ((len(categorical_cols) + 2) // 3)))
            for i, col in enumerate(categorical_cols, 1):
                ax = plt.subplot((len(categorical_cols) + 2) // 3, 3, i)
                
                # Count plot with percentage
                cat_counts = self.df[col].value_counts()
                total = len(self.df)
                
                sns.barplot(
                    x=cat_counts.index, 
                    y=cat_counts.values, 
                    palette='Set3'
                )
                
                # Add percentage labels on bars
                for j, v in enumerate(cat_counts.values):
                    percentage = v / total * 100
                    ax.text(
                        j, v, f'{percentage:.1f}%', 
                        ha='center', va='bottom',
                        fontweight='bold'
                    )
                
                plt.title(f'Distribution of {col}', fontweight='bold')
                plt.xlabel(col)
                plt.ylabel('Count')
                plt.xticks(rotation=45, ha='right')
            
            plt.suptitle('Categorical Column Distributions', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'categorical_distributions.png'), 
                        dpi=300, bbox_inches='tight')
            plt.close()

        print("✅ Visualizations Generated with Comprehensive Labeling!")

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
        {json.dumps(analysis_details['missing_values'], indent=2)}
        
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

    def generate_readme(self, narrative: str, data_story: str):
        """
        Generate comprehensive README with story-first approach
        
        Args:
            narrative (str): Original LLM analysis narrative
            data_story (str): Generated data story
        """
        readme_path = os.path.join(self.output_dir, 'README.md')
        with open(readme_path, 'w', encoding='utf-8') as f:
            # 🌟 Story-First Approach 🌟
            f.write("# 📖 Data Story: Unveiling Hidden Narratives\n\n")
            f.write(data_story + "\n\n")
            
            # Add a visual separator
            f.write("---\n\n")
            
            # Add an attention-grabbing intro
            f.write("## 🔍 Dive Deeper: Comprehensive Data Analysis\n\n")
            f.write("*The story above is just the beginning. Let's explore the data in detail...*\n\n")
            
            # Original narrative follows
            f.write("### Detailed Analysis Insights\n\n")
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
        
        # Generate Story (New Feature)
        try:
            storyteller = DataStoryteller(self.df)
            data_story = storyteller.generate_story(self.api_token)
            
            # Generate README with story-first approach
            self.generate_readme(narrative, data_story)
        except Exception as e:
            # Fallback if story generation fails
            self.generate_readme(narrative, 
                "## 🚨 Story Generation Unavailable\n\n"
                "While we couldn't generate a narrative this time, "
                "our detailed analysis provides valuable insights."
            )
            print(f"❗ Story generation failed: {e}")
        
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
        print(f"Numeric Columns: {', '.join (results['numeric_columns'])}")
        print(f"Categorical Columns: {', '.join(results['categorical_columns'])}")
    
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()





