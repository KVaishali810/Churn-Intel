# ============================================================
# Customer Churn Prediction - Minor Project
# Tech Stack: Python, Pandas, Scikit-learn, Logistic Regression
# ============================================================

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================
# 1. GENERATE SYNTHETIC CUSTOMER DATA
# ============================================================
n = 1000

tenure         = np.random.randint(1, 72, n)
monthly_charge = np.round(np.random.uniform(20, 120, n), 2)
total_charges  = np.round(monthly_charge * tenure + np.random.normal(0, 50, n), 2)
total_charges  = np.clip(total_charges, 0, None)
num_products   = np.random.randint(1, 5, n)
support_calls  = np.random.poisson(2, n)
contract_type  = np.random.choice(['Month-to-Month', 'One Year', 'Two Year'],
                                   n, p=[0.55, 0.25, 0.20])
payment_method = np.random.choice(['Electronic Check', 'Mailed Check',
                                   'Bank Transfer', 'Credit Card'], n)
internet_service = np.random.choice(['DSL', 'Fiber Optic', 'No'], n, p=[0.35, 0.45, 0.20])
gender         = np.random.choice(['Male', 'Female'], n)
senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])

# Churn probability based on real-world logic
churn_prob = (
    0.3
    - 0.004 * tenure
    + 0.003 * monthly_charge
    - 0.05  * num_products
    + 0.04  * support_calls
    + 0.15  * (contract_type == 'Month-to-Month')
    - 0.1   * (contract_type == 'Two Year')
    + 0.08  * (internet_service == 'Fiber Optic')
    + 0.05  * senior_citizen
    + np.random.normal(0, 0.05, n)
)
churn_prob = np.clip(churn_prob, 0.02, 0.98)
churn      = (np.random.rand(n) < churn_prob).astype(int)

df = pd.DataFrame({
    'CustomerID'     : [f'CUST{str(i).zfill(4)}' for i in range(1, n+1)],
    'Gender'         : gender,
    'SeniorCitizen'  : senior_citizen,
    'Tenure'         : tenure,
    'NumProducts'    : num_products,
    'InternetService': internet_service,
    'ContractType'   : contract_type,
    'PaymentMethod'  : payment_method,
    'MonthlyCharges' : monthly_charge,
    'TotalCharges'   : total_charges,
    'SupportCalls'   : support_calls,
    'Churn'          : churn
})

df.to_csv('/home/claude/churn_project/customer_data.csv', index=False)
print("✅ Dataset created:", df.shape)
print(df.head(3))

# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================
churn_rate = df['Churn'].mean() * 100
print(f"\n📊 Overall Churn Rate: {churn_rate:.1f}%")
print(df.describe().round(2))

# ============================================================
# 3. PREPROCESSING
# ============================================================
df_model = df.drop(columns=['CustomerID'])

# Encode categoricals
df_model = pd.get_dummies(df_model,
    columns=['Gender','InternetService','ContractType','PaymentMethod'],
    drop_first=True)

X = df_model.drop('Churn', axis=1)
y = df_model['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# ============================================================
# 4. MODEL TRAINING — Logistic Regression Pipeline
# ============================================================
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  LogisticRegression(max_iter=500, random_state=42, C=1.0))
])
pipeline.fit(X_train, y_train)

y_pred  = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:,1]

acc     = accuracy_score(y_test, y_pred)
auc     = roc_auc_score(y_test, y_proba)
print(f"\n✅ Accuracy : {acc:.4f}")
print(f"✅ ROC-AUC  : {auc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ============================================================
# 5. VISUALISATIONS  (saved to PNG)
# ============================================================
palette = {'churn': '#E74C3C', 'stay': '#2ECC71', 'neutral': '#3498DB',
           'bg': '#F8F9FA', 'dark': '#2C3E50'}

fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor(palette['bg'])
gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

# -- (A) Churn Distribution
ax1 = fig.add_subplot(gs[0, 0])
labels = ['Stayed', 'Churned']
sizes  = [y.value_counts()[0], y.value_counts()[1]]
colors = [palette['stay'], palette['churn']]
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors,
        startangle=90, textprops={'fontsize':10, 'fontweight':'bold'})
ax1.set_title('Churn Distribution', fontweight='bold', color=palette['dark'])

# -- (B) Churn by Contract Type
ax2 = fig.add_subplot(gs[0, 1])
ct_churn = df.groupby('ContractType')['Churn'].mean() * 100
bars = ax2.bar(ct_churn.index, ct_churn.values,
               color=[palette['churn'], palette['neutral'], palette['stay']])
ax2.set_title('Churn Rate by Contract Type', fontweight='bold', color=palette['dark'])
ax2.set_ylabel('Churn Rate (%)')
ax2.set_ylim(0, 80)
for bar, val in zip(bars, ct_churn.values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+1,
             f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
ax2.tick_params(axis='x', rotation=15)

# -- (C) Tenure Distribution by Churn
ax3 = fig.add_subplot(gs[0, 2])
for label, grp, col in [(0,'Stayed',palette['stay']),(1,'Churned',palette['churn'])]:
    ax3.hist(df[df['Churn']==label]['Tenure'], bins=20, alpha=0.6,
             color=col, label=grp, edgecolor='white')
ax3.set_title('Tenure Distribution by Churn', fontweight='bold', color=palette['dark'])
ax3.set_xlabel('Tenure (months)')
ax3.set_ylabel('Count')
ax3.legend()

# -- (D) Monthly Charges by Churn
ax4 = fig.add_subplot(gs[1, 0])
ax4.boxplot([df[df['Churn']==0]['MonthlyCharges'],
             df[df['Churn']==1]['MonthlyCharges']],
            labels=['Stayed','Churned'],
            patch_artist=True,
            boxprops=dict(facecolor=palette['neutral'], alpha=0.6))
ax4.set_title('Monthly Charges vs Churn', fontweight='bold', color=palette['dark'])
ax4.set_ylabel('Monthly Charges ($)')

# -- (E) Support Calls vs Churn
ax5 = fig.add_subplot(gs[1, 1])
sc_churn = df.groupby('SupportCalls')['Churn'].mean() * 100
ax5.bar(sc_churn.index, sc_churn.values, color=palette['churn'], alpha=0.8, edgecolor='white')
ax5.set_title('Churn Rate by Support Calls', fontweight='bold', color=palette['dark'])
ax5.set_xlabel('Number of Support Calls')
ax5.set_ylabel('Churn Rate (%)')

# -- (F) Feature Importance (Coefficients)
ax6 = fig.add_subplot(gs[1, 2])
coefs = pipeline.named_steps['model'].coef_[0]
feat_names = X.columns.tolist()
coef_df = pd.DataFrame({'Feature': feat_names, 'Coefficient': coefs})
coef_df = coef_df.reindex(coef_df['Coefficient'].abs().sort_values(ascending=True).index)
top10 = coef_df.tail(10)
colors_bar = [palette['churn'] if c > 0 else palette['stay'] for c in top10['Coefficient']]
ax6.barh(top10['Feature'], top10['Coefficient'], color=colors_bar, edgecolor='white')
ax6.axvline(0, color='black', linewidth=0.8)
ax6.set_title('Top Feature Coefficients', fontweight='bold', color=palette['dark'])
ax6.set_xlabel('Coefficient Value')
ax6.tick_params(axis='y', labelsize=7)

# -- (G) Confusion Matrix
ax7 = fig.add_subplot(gs[2, 0])
cm = confusion_matrix(y_test, y_pred)
im = ax7.imshow(cm, interpolation='nearest', cmap='RdYlGn')
ax7.set_title('Confusion Matrix', fontweight='bold', color=palette['dark'])
ax7.set_xticks([0,1]); ax7.set_yticks([0,1])
ax7.set_xticklabels(['Stayed','Churned']); ax7.set_yticklabels(['Stayed','Churned'])
ax7.set_xlabel('Predicted'); ax7.set_ylabel('Actual')
for i in range(2):
    for j in range(2):
        ax7.text(j, i, str(cm[i,j]), ha='center', va='center',
                 fontsize=16, fontweight='bold', color='white')

# -- (H) ROC Curve
ax8 = fig.add_subplot(gs[2, 1])
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax8.plot(fpr, tpr, color=palette['churn'], lw=2, label=f'AUC = {auc:.3f}')
ax8.plot([0,1],[0,1], 'k--', lw=1)
ax8.fill_between(fpr, tpr, alpha=0.1, color=palette['churn'])
ax8.set_title('ROC Curve', fontweight='bold', color=palette['dark'])
ax8.set_xlabel('False Positive Rate'); ax8.set_ylabel('True Positive Rate')
ax8.legend(loc='lower right')

# -- (I) Metrics Summary
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
from sklearn.metrics import precision_score, recall_score, f1_score
metrics = {
    'Accuracy' : f'{acc*100:.1f}%',
    'ROC-AUC'  : f'{auc:.3f}',
    'Precision': f'{precision_score(y_test,y_pred)*100:.1f}%',
    'Recall'   : f'{recall_score(y_test,y_pred)*100:.1f}%',
    'F1-Score' : f'{f1_score(y_test,y_pred)*100:.1f}%',
    'Train Size': f'{len(X_train)} samples',
    'Test Size' : f'{len(X_test)} samples',
    'Features'  : f'{X.shape[1]} features',
}
y_pos = 0.95
ax9.text(0.5, 1.05, '📋 Model Summary', ha='center', va='top',
         fontsize=12, fontweight='bold', transform=ax9.transAxes, color=palette['dark'])
for k, v in metrics.items():
    color = palette['churn'] if k in ['Accuracy','ROC-AUC','F1-Score'] else palette['dark']
    ax9.text(0.05, y_pos, f'{k}:', transform=ax9.transAxes,
             fontsize=10, fontweight='bold', color=palette['dark'])
    ax9.text(0.6, y_pos, v, transform=ax9.transAxes,
             fontsize=10, fontweight='bold', color=color)
    y_pos -= 0.11

fig.suptitle('Customer Churn Prediction Dashboard\nLogistic Regression Model',
             fontsize=16, fontweight='bold', color=palette['dark'], y=1.01)

plt.savefig('/home/claude/churn_project/churn_dashboard.png',
            dpi=150, bbox_inches='tight', facecolor=palette['bg'])
plt.close()
print("\n✅ Dashboard saved!")

# ============================================================
# 6. SAVE PREDICTIONS
# ============================================================
results = X_test.copy()
results['Actual_Churn']      = y_test.values
results['Predicted_Churn']   = y_pred
results['Churn_Probability'] = np.round(y_proba, 4)
results.to_csv('/home/claude/churn_project/predictions.csv')
print("✅ Predictions saved!")
