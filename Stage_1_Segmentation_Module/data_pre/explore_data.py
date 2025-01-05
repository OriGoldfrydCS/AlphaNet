import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# File path to the CSV dataset that contains the "sentence" column
file_path = 'Stage_1_Segmentation_Module/data/sentences_data_5000.csv'

# Read the CSV file into a pandas DataFrame
data = pd.read_csv(file_path)

# Preprocess the data:
# 1. "Sentence Length": the number of words in each sentence.
#    - We split on whitespace and count the resulting tokens.
# 2. "Number of Letters and Chars": total alphanumeric characters in the sentence.
#    - We use a generator expression to test each character with c.isalnum().
data['Sentence Length'] = data['sentence'].apply(lambda x: len(x.split()))
data['Number of Letters and Chars'] = data['sentence'].apply(lambda x: sum(c.isalnum() for c in x))

# Group by "Sentence Length" and:
#  - Sum the "Number of Letters and Chars" for each group
#  - Count how many sentences fall into that group
summary = data.groupby('Sentence Length').agg(
    Total_Letters_Chars=('Number of Letters and Chars', 'sum'),
    Sentence_Count=('Sentence Length', 'count')
).reset_index()

# Calculate the percentage of total letters/chars contributed by each
# "Sentence Length" group compared to the grand total of letters/chars
summary['Percentage (%)'] = ((summary['Total_Letters_Chars'] / summary['Total_Letters_Chars'].sum()) * 100).apply(lambda x: f"{x:.2f}")

# Sort by Sentence Length
summary = summary.sort_values(by='Sentence Length', ascending=True)

# Create a Plotly table
fig_table = go.Figure(data=[
    go.Table(
        header=dict(
            values=["<b>Sentence Length</b>", "<b>Sentence Count</b>", "<b>Total Letters and Chars</b>", "<b>Percentage (%)</b>"],
            fill_color='lightblue',
            align='left'
        ),
        cells=dict(
            values=[
                summary['Sentence Length'].astype(str),
                summary['Sentence_Count'].astype(str),
                summary['Total_Letters_Chars'].astype(str),
                summary['Percentage (%)'].astype(str)
            ],
            fill_color='lavender',
            align='left'
        )
    )
])

# Show the table
fig_table.show()

# Create a matplotlib bar plot for percentage distribution
plt.figure(figsize=(10, 6))
plt.bar(summary['Sentence Length'], summary['Percentage (%)'].astype(float), color='skyblue')
plt.title('Percentage Distribution of Sentence Lengths', fontsize=14)
plt.xlabel('Sentence Length (Number of Words)', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(ticks=range(1, min(11, summary['Sentence Length'].max() + 1)), fontsize=10)
plt.yticks(fontsize=10)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
