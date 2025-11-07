"""
Advanced Visualization Suite for Mental Health & Social Media Analysis
=======================================================================
Creates publication-ready, portfolio-quality visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('data/Mental_Health_and_Social_Media_Balance_Dataset.csv')

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)
colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c']

# ==========================================
# VISUALIZATION 1: Executive Dashboard
# ==========================================

fig = plt.figure(figsize=(20, 12))
gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

# 1. Correlation Heatmap
ax1 = fig.add_subplot(gs[0, :2])
numerical_cols = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 
                  'Stress_Level(1-10)', 'Days_Without_Social_Media',
                  'Exercise_Frequency(week)', 'Happiness_Index(1-10)']
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, ax=ax1, cbar_kws={"shrink": 0.8})
ax1.set_title('Correlation Matrix: All Variables', fontsize=16, fontweight='bold', pad=20)

# 2. Happiness Distribution
ax2 = fig.add_subplot(gs[0, 2])
df['Happiness_Index(1-10)'].hist(bins=10, color='#2ecc71', edgecolor='black', ax=ax2, alpha=0.7)
ax2.axvline(df['Happiness_Index(1-10)'].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
ax2.set_xlabel('Happiness Index', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Happiness Distribution', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(alpha=0.3)

# 3. Screen Time vs Happiness (Scatter with regression)
ax3 = fig.add_subplot(gs[1, 0])
ax3.scatter(df['Daily_Screen_Time(hrs)'], df['Happiness_Index(1-10)'], 
           alpha=0.4, c=df['Stress_Level(1-10)'], cmap='RdYlGn_r', s=50)
z = np.polyfit(df['Daily_Screen_Time(hrs)'], df['Happiness_Index(1-10)'], 2)
p = np.poly1d(z)
x_line = np.linspace(df['Daily_Screen_Time(hrs)'].min(), df['Daily_Screen_Time(hrs)'].max(), 100)
ax3.plot(x_line, p(x_line), "r--", linewidth=2, label='Trend')
ax3.set_xlabel('Daily Screen Time (hours)', fontsize=12)
ax3.set_ylabel('Happiness Index', fontsize=12)
ax3.set_title('Screen Time vs Happiness', fontsize=14, fontweight='bold')
ax3.legend()
ax3.grid(alpha=0.3)

# 4. Platform Comparison
ax4 = fig.add_subplot(gs[1, 1])
platform_happiness = df.groupby('Social_Media_Platform')['Happiness_Index(1-10)'].mean().sort_values()
platform_happiness.plot(kind='barh', ax=ax4, color=colors)
ax4.set_xlabel('Average Happiness', fontsize=12)
ax4.set_title('Platform Comparison', fontsize=14, fontweight='bold')
ax4.grid(axis='x', alpha=0.3)

# 5. Sleep Quality Impact
ax5 = fig.add_subplot(gs[1, 2])
sleep_groups = df.groupby('Sleep_Quality(1-10)')['Happiness_Index(1-10)'].mean()
ax5.plot(sleep_groups.index, sleep_groups.values, marker='o', linewidth=3, 
         markersize=8, color='#3498db')
ax5.fill_between(sleep_groups.index, sleep_groups.values, alpha=0.3, color='#3498db')
ax5.set_xlabel('Sleep Quality', fontsize=12)
ax5.set_ylabel('Average Happiness', fontsize=12)
ax5.set_title('Sleep Quality Impact', fontsize=14, fontweight='bold')
ax5.grid(alpha=0.3)

# 6. Age Group Analysis
ax6 = fig.add_subplot(gs[2, 0])
df['Age_Group'] = pd.cut(df['Age'], bins=[15, 20, 30, 40, 50],
                         labels=['16-20', '21-30', '31-40', '41-50'])
age_data = df.groupby('Age_Group').agg({
    'Happiness_Index(1-10)': 'mean',
    'Stress_Level(1-10)': 'mean'
})
x = np.arange(len(age_data))
width = 0.35
ax6.bar(x - width/2, age_data['Happiness_Index(1-10)'], width, label='Happiness', color='#2ecc71')
ax6.bar(x + width/2, age_data['Stress_Level(1-10)'], width, label='Stress', color='#e74c3c')
ax6.set_xlabel('Age Group', fontsize=12)
ax6.set_ylabel('Score (1-10)', fontsize=12)
ax6.set_title('Age Group Comparison', fontsize=14, fontweight='bold')
ax6.set_xticks(x)
ax6.set_xticklabels(age_data.index)
ax6.legend()
ax6.grid(axis='y', alpha=0.3)

# 7. Exercise Frequency Impact
ax7 = fig.add_subplot(gs[2, 1])
exercise_happiness = df.groupby('Exercise_Frequency(week)')['Happiness_Index(1-10)'].mean()
ax7.bar(exercise_happiness.index, exercise_happiness.values, color='#9b59b6', alpha=0.7, edgecolor='black')
ax7.set_xlabel('Exercise Days/Week', fontsize=12)
ax7.set_ylabel('Average Happiness', fontsize=12)
ax7.set_title('Exercise Impact on Happiness', fontsize=14, fontweight='bold')
ax7.grid(axis='y', alpha=0.3)

# 8. Gender Comparison
ax8 = fig.add_subplot(gs[2, 2])
gender_data = df.groupby('Gender').agg({
    'Happiness_Index(1-10)': 'mean',
    'Daily_Screen_Time(hrs)': 'mean',
    'Stress_Level(1-10)': 'mean'
})
gender_data.plot(kind='bar', ax=ax8, color=['#2ecc71', '#e74c3c', '#f39c12'], alpha=0.7)
ax8.set_xlabel('Gender', fontsize=12)
ax8.set_ylabel('Score', fontsize=12)
ax8.set_title('Gender-based Metrics', fontsize=14, fontweight='bold')
ax8.legend(loc='upper right', fontsize=9)
ax8.set_xticklabels(gender_data.index, rotation=0)
ax8.grid(axis='y', alpha=0.3)

plt.suptitle('Mental Health & Social Media: Executive Dashboard', 
             fontsize=20, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('executive_dashboard.png', dpi=300, bbox_inches='tight')
print("✅ Executive Dashboard saved as 'executive_dashboard.png'")

# ==========================================
# VISUALIZATION 2: Screen Time Deep Dive
# ==========================================

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Create screen time categories
df['Screen_Time_Category'] = pd.cut(df['Daily_Screen_Time(hrs)'], 
                                     bins=[0, 3, 5, 7, 9, 12],
                                     labels=['1-3hrs', '3-5hrs', '5-7hrs', '7-9hrs', '9+hrs'])

# 1. Screen Time Distribution
screen_time_counts = df['Screen_Time_Category'].value_counts().sort_index()
axes[0, 0].bar(range(len(screen_time_counts)), screen_time_counts.values, 
               color=colors[:len(screen_time_counts)], alpha=0.7, edgecolor='black')
axes[0, 0].set_xticks(range(len(screen_time_counts)))
axes[0, 0].set_xticklabels(screen_time_counts.index, rotation=45)
axes[0, 0].set_ylabel('Number of Users', fontsize=12)
axes[0, 0].set_title('Screen Time Distribution', fontsize=14, fontweight='bold')
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Screen Time vs Wellbeing Metrics
screen_metrics = df.groupby('Screen_Time_Category').agg({
    'Happiness_Index(1-10)': 'mean',
    'Sleep_Quality(1-10)': 'mean',
    'Stress_Level(1-10)': 'mean'
})
screen_metrics.plot(ax=axes[0, 1], marker='o', linewidth=2, markersize=8)
axes[0, 1].set_xlabel('Screen Time Category', fontsize=12)
axes[0, 1].set_ylabel('Score (1-10)', fontsize=12)
axes[0, 1].set_title('Screen Time Impact on Wellbeing', fontsize=14, fontweight='bold')
axes[0, 1].legend(loc='best')
axes[0, 1].grid(alpha=0.3)
axes[0, 1].set_xticks(range(len(screen_metrics.index)))
axes[0, 1].set_xticklabels(screen_metrics.index, rotation=45)

# 3. Box plot - Screen Time vs Happiness
df.boxplot(column='Happiness_Index(1-10)', by='Screen_Time_Category', ax=axes[1, 0])
axes[1, 0].set_xlabel('Screen Time Category', fontsize=12)
axes[1, 0].set_ylabel('Happiness Index', fontsize=12)
axes[1, 0].set_title('Happiness Variability by Screen Time', fontsize=14, fontweight='bold')
plt.sca(axes[1, 0])
plt.xticks(rotation=45)
axes[1, 0].get_figure().suptitle('')  # Remove default title

# 4. Screen Time by Platform
platform_screen = df.groupby('Social_Media_Platform')['Daily_Screen_Time(hrs)'].mean().sort_values()
axes[1, 1].barh(range(len(platform_screen)), platform_screen.values, color=colors)
axes[1, 1].set_yticks(range(len(platform_screen)))
axes[1, 1].set_yticklabels(platform_screen.index)
axes[1, 1].set_xlabel('Average Daily Screen Time (hours)', fontsize=12)
axes[1, 1].set_title('Screen Time by Platform', fontsize=14, fontweight='bold')
axes[1, 1].grid(axis='x', alpha=0.3)

plt.suptitle('Screen Time Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('screen_time_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Screen Time Analysis saved as 'screen_time_analysis.png'")

# ==========================================
# VISUALIZATION 3: Interactive Plotly Dashboard
# ==========================================

# Create subplot figure
fig = make_subplots(
    rows=3, cols=2,
    subplot_titles=('Happiness by Platform', 'Screen Time Distribution',
                    'Sleep vs Stress vs Happiness', 'Age Distribution',
                    'Exercise Impact', 'Gender Comparison'),
    specs=[[{"type": "bar"}, {"type": "histogram"}],
           [{"type": "scatter3d", "colspan": 2}, None],
           [{"type": "box"}, {"type": "bar"}]],
    vertical_spacing=0.12,
    horizontal_spacing=0.1
)

# 1. Happiness by Platform
platform_stats = df.groupby('Social_Media_Platform')['Happiness_Index(1-10)'].mean().sort_values()
fig.add_trace(
    go.Bar(x=platform_stats.index, y=platform_stats.values, 
           marker_color='lightblue', name='Avg Happiness'),
    row=1, col=1
)

# 2. Screen Time Histogram
fig.add_trace(
    go.Histogram(x=df['Daily_Screen_Time(hrs)'], nbinsx=20,
                 marker_color='coral', name='Screen Time'),
    row=1, col=2
)

# 3. 3D Scatter Plot
fig.add_trace(
    go.Scatter3d(
        x=df['Sleep_Quality(1-10)'],
        y=df['Stress_Level(1-10)'],
        z=df['Happiness_Index(1-10)'],
        mode='markers',
        marker=dict(
            size=4,
            color=df['Happiness_Index(1-10)'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Happiness")
        ),
        name='Users'
    ),
    row=2, col=1
)

# 4. Age Distribution (moved to box plot position)
fig.add_trace(
    go.Box(y=df['Age'], marker_color='lightgreen', name='Age'),
    row=3, col=1
)

# 5. Exercise Impact
exercise_data = df.groupby('Exercise_Frequency(week)')['Happiness_Index(1-10)'].mean()
fig.add_trace(
    go.Bar(x=exercise_data.index, y=exercise_data.values,
           marker_color='mediumpurple', name='Exercise Impact'),
    row=3, col=2
)

# Update layout
fig.update_layout(
    title_text="Interactive Mental Health & Social Media Dashboard",
    title_font_size=24,
    showlegend=False,
    height=1200,
    template='plotly_white'
)

fig.update_xaxes(title_text="Platform", row=1, col=1)
fig.update_xaxes(title_text="Screen Time (hrs)", row=1, col=2)
fig.update_xaxes(title_text="Sleep Quality", row=2, col=1)
fig.update_xaxes(title_text="Exercise Days/Week", row=3, col=2)

fig.update_yaxes(title_text="Happiness", row=1, col=1)
fig.update_yaxes(title_text="Count", row=1, col=2)
fig.update_yaxes(title_text="Age", row=3, col=1)
fig.update_yaxes(title_text="Happiness", row=3, col=2)

fig.write_html('interactive_dashboard.html')
print("✅ Interactive Dashboard saved as 'interactive_dashboard.html'")

# ==========================================
# VISUALIZATION 4: Feature Importance Plot
# ==========================================

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Prepare data for ML
feature_cols = ['Age', 'Daily_Screen_Time(hrs)', 'Sleep_Quality(1-10)', 
                'Stress_Level(1-10)', 'Days_Without_Social_Media',
                'Exercise_Frequency(week)']

X = df[feature_cols].copy()
y = df['Happiness_Index(1-10)']

# Add encoded features
le_gender = LabelEncoder()
le_platform = LabelEncoder()
X['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
X['Platform_Encoded'] = le_platform.fit_transform(df['Social_Media_Platform'])

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Get feature importance
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=True)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors[:len(importance_df)])
ax.set_xlabel('Importance Score', fontsize=14, fontweight='bold')
ax.set_title('ML Feature Importance: Happiness Predictors', fontsize=16, fontweight='bold', pad=20)
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, bar in enumerate(bars):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2, 
            f'{width:.3f}', ha='left', va='center', fontsize=10)

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✅ Feature Importance saved as 'feature_importance.png'")

# ==========================================
# VISUALIZATION 5: Segment Analysis
# ==========================================

# Create wellbeing score and segments
df['Wellbeing_Score'] = (
    df['Sleep_Quality(1-10)'] * 0.3 +
    (10 - df['Stress_Level(1-10)']) * 0.3 +
    df['Happiness_Index(1-10)'] * 0.2 +
    (10 - df['Daily_Screen_Time(hrs)']) * 0.1 +
    df['Exercise_Frequency(week)'] * 0.1
)

df['Segment'] = pd.cut(df['Wellbeing_Score'], 
                       bins=[0, 5, 6.5, 8, 10],
                       labels=['Critical', 'At Risk', 'Balanced', 'Thriving'])

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Segment Distribution
segment_counts = df['Segment'].value_counts()
colors_seg = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']
wedges, texts, autotexts = axes[0, 0].pie(segment_counts.values, labels=segment_counts.index,
                                           autopct='%1.1f%%', startangle=90, colors=colors_seg)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontsize(12)
    autotext.set_fontweight('bold')
axes[0, 0].set_title('User Segment Distribution', fontsize=14, fontweight='bold')

# 2. Segment Characteristics
segment_chars = df.groupby('Segment').agg({
    'Happiness_Index(1-10)': 'mean',
    'Daily_Screen_Time(hrs)': 'mean',
    'Sleep_Quality(1-10)': 'mean',
    'Stress_Level(1-10)': 'mean'
})
segment_chars.T.plot(kind='bar', ax=axes[0, 1], color=colors_seg, alpha=0.7)
axes[0, 1].set_title('Segment Characteristics', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Metric', fontsize=12)
axes[0, 1].set_ylabel('Score', fontsize=12)
axes[0, 1].legend(title='Segment', loc='upper right')
axes[0, 1].grid(axis='y', alpha=0.3)
axes[0, 1].set_xticklabels(segment_chars.T.index, rotation=45, ha='right')

# 3. Segment by Age Group
segment_age = pd.crosstab(df['Age_Group'], df['Segment'], normalize='index') * 100
segment_age.plot(kind='bar', stacked=True, ax=axes[1, 0], color=colors_seg, alpha=0.7)
axes[1, 0].set_title('Segment Distribution by Age Group', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Age Group', fontsize=12)
axes[1, 0].set_ylabel('Percentage (%)', fontsize=12)
axes[1, 0].legend(title='Segment', loc='upper right')
axes[1, 0].set_xticklabels(segment_age.index, rotation=0)
axes[1, 0].grid(axis='y', alpha=0.3)

# 4. Segment by Platform
segment_platform = df.groupby(['Social_Media_Platform', 'Segment']).size().unstack(fill_value=0)
segment_platform.plot(kind='bar', ax=axes[1, 1], color=colors_seg, alpha=0.7)
axes[1, 1].set_title('Segment Distribution by Platform', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Platform', fontsize=12)
axes[1, 1].set_ylabel('Count', fontsize=12)
axes[1, 1].legend(title='Segment', loc='upper right')
axes[1, 1].set_xticklabels(segment_platform.index, rotation=45, ha='right')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.suptitle('User Segmentation Analysis', fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('segment_analysis.png', dpi=300, bbox_inches='tight')
print("✅ Segment Analysis saved as 'segment_analysis.png'")

# ==========================================
# SUMMARY
# ==========================================

print("\n" + "="*80)
print("📊 VISUALIZATION SUITE COMPLETE")
print("="*80)
print("\nGenerated Files:")
print("1. executive_dashboard.png - Comprehensive overview (20x12)")
print("2. screen_time_analysis.png - Deep dive into screen time effects")
print("3. interactive_dashboard.html - Interactive Plotly dashboard")
print("4. feature_importance.png - ML model insights")
print("5. segment_analysis.png - User segmentation visuals")
print("\n💡 Use these for:")
print("   • Portfolio presentations")
print("   • GitHub README")
print("   • Medium/Blog articles")
print("   • LinkedIn posts")
print("   • Interview presentations")
print("\n✅ All visualizations are publication-ready at 300 DPI")
print("="*80)