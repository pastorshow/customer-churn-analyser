{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "e4065e12-6dda-4565-bc42-5bda1e690a4d",
   "metadata": {},
   "source": [
    "Customer Churn Prediction"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "0d95855d-7e16-469e-aa3a-048417538261",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerID</th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Usage Frequency</th>\n",
       "      <th>Support Calls</th>\n",
       "      <th>Payment Delay</th>\n",
       "      <th>Subscription Type</th>\n",
       "      <th>Contract Length</th>\n",
       "      <th>Total Spend</th>\n",
       "      <th>Last Interaction</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>22</td>\n",
       "      <td>Female</td>\n",
       "      <td>25</td>\n",
       "      <td>14</td>\n",
       "      <td>4</td>\n",
       "      <td>27</td>\n",
       "      <td>Basic</td>\n",
       "      <td>Monthly</td>\n",
       "      <td>598</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>2</td>\n",
       "      <td>41</td>\n",
       "      <td>Female</td>\n",
       "      <td>28</td>\n",
       "      <td>28</td>\n",
       "      <td>7</td>\n",
       "      <td>13</td>\n",
       "      <td>Standard</td>\n",
       "      <td>Monthly</td>\n",
       "      <td>584</td>\n",
       "      <td>20</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3</td>\n",
       "      <td>47</td>\n",
       "      <td>Male</td>\n",
       "      <td>27</td>\n",
       "      <td>10</td>\n",
       "      <td>2</td>\n",
       "      <td>29</td>\n",
       "      <td>Premium</td>\n",
       "      <td>Annual</td>\n",
       "      <td>757</td>\n",
       "      <td>21</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>4</td>\n",
       "      <td>35</td>\n",
       "      <td>Male</td>\n",
       "      <td>9</td>\n",
       "      <td>12</td>\n",
       "      <td>5</td>\n",
       "      <td>17</td>\n",
       "      <td>Premium</td>\n",
       "      <td>Quarterly</td>\n",
       "      <td>232</td>\n",
       "      <td>18</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>5</td>\n",
       "      <td>53</td>\n",
       "      <td>Female</td>\n",
       "      <td>58</td>\n",
       "      <td>24</td>\n",
       "      <td>9</td>\n",
       "      <td>2</td>\n",
       "      <td>Standard</td>\n",
       "      <td>Annual</td>\n",
       "      <td>533</td>\n",
       "      <td>18</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   CustomerID  Age  Gender  Tenure  Usage Frequency  Support Calls  \\\n",
       "0           1   22  Female      25               14              4   \n",
       "1           2   41  Female      28               28              7   \n",
       "2           3   47    Male      27               10              2   \n",
       "3           4   35    Male       9               12              5   \n",
       "4           5   53  Female      58               24              9   \n",
       "\n",
       "   Payment Delay Subscription Type Contract Length  Total Spend  \\\n",
       "0             27             Basic         Monthly          598   \n",
       "1             13          Standard         Monthly          584   \n",
       "2             29           Premium          Annual          757   \n",
       "3             17           Premium       Quarterly          232   \n",
       "4              2          Standard          Annual          533   \n",
       "\n",
       "   Last Interaction  Churn  \n",
       "0                 9      1  \n",
       "1                20      0  \n",
       "2                21      0  \n",
       "3                18      0  \n",
       "4                18      0  "
      ]
     },
     "execution_count": 1,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import (\n",
    "    classification_report, confusion_matrix,\n",
    "    roc_auc_score, roc_curve, ConfusionMatrixDisplay\n",
    ")\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    " \n",
    "\n",
    "df = pd.read_csv(r'C:\\Users\\PSALMS COMPUTER\\Desktop\\customer_churn_dataset-testing-master.csv')\n",
    "\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "a9dfdfaa-44fc-4fe6-80e7-ef6134fd8041",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>CustomerID</th>\n",
       "      <th>Age</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Tenure</th>\n",
       "      <th>Usage Frequency</th>\n",
       "      <th>Support Calls</th>\n",
       "      <th>Payment Delay</th>\n",
       "      <th>Subscription Type</th>\n",
       "      <th>Contract Length</th>\n",
       "      <th>Total Spend</th>\n",
       "      <th>Last Interaction</th>\n",
       "      <th>Churn</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>64369</th>\n",
       "      <td>64370</td>\n",
       "      <td>45</td>\n",
       "      <td>Female</td>\n",
       "      <td>33</td>\n",
       "      <td>12</td>\n",
       "      <td>6</td>\n",
       "      <td>21</td>\n",
       "      <td>Basic</td>\n",
       "      <td>Quarterly</td>\n",
       "      <td>947</td>\n",
       "      <td>14</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64370</th>\n",
       "      <td>64371</td>\n",
       "      <td>37</td>\n",
       "      <td>Male</td>\n",
       "      <td>6</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>22</td>\n",
       "      <td>Standard</td>\n",
       "      <td>Annual</td>\n",
       "      <td>923</td>\n",
       "      <td>9</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64371</th>\n",
       "      <td>64372</td>\n",
       "      <td>25</td>\n",
       "      <td>Male</td>\n",
       "      <td>39</td>\n",
       "      <td>14</td>\n",
       "      <td>8</td>\n",
       "      <td>30</td>\n",
       "      <td>Premium</td>\n",
       "      <td>Monthly</td>\n",
       "      <td>327</td>\n",
       "      <td>20</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64372</th>\n",
       "      <td>64373</td>\n",
       "      <td>50</td>\n",
       "      <td>Female</td>\n",
       "      <td>18</td>\n",
       "      <td>19</td>\n",
       "      <td>7</td>\n",
       "      <td>22</td>\n",
       "      <td>Standard</td>\n",
       "      <td>Monthly</td>\n",
       "      <td>540</td>\n",
       "      <td>13</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>64373</th>\n",
       "      <td>64374</td>\n",
       "      <td>52</td>\n",
       "      <td>Female</td>\n",
       "      <td>45</td>\n",
       "      <td>15</td>\n",
       "      <td>9</td>\n",
       "      <td>25</td>\n",
       "      <td>Standard</td>\n",
       "      <td>Monthly</td>\n",
       "      <td>696</td>\n",
       "      <td>22</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "       CustomerID  Age  Gender  Tenure  Usage Frequency  Support Calls  \\\n",
       "64369       64370   45  Female      33               12              6   \n",
       "64370       64371   37    Male       6                1              5   \n",
       "64371       64372   25    Male      39               14              8   \n",
       "64372       64373   50  Female      18               19              7   \n",
       "64373       64374   52  Female      45               15              9   \n",
       "\n",
       "       Payment Delay Subscription Type Contract Length  Total Spend  \\\n",
       "64369             21             Basic       Quarterly          947   \n",
       "64370             22          Standard          Annual          923   \n",
       "64371             30           Premium         Monthly          327   \n",
       "64372             22          Standard         Monthly          540   \n",
       "64373             25          Standard         Monthly          696   \n",
       "\n",
       "       Last Interaction  Churn  \n",
       "64369                14      1  \n",
       "64370                 9      1  \n",
       "64371                20      1  \n",
       "64372                13      1  \n",
       "64373                22      1  "
      ]
     },
     "execution_count": 2,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "f29d228b-4deb-4091-81a8-a8a04fb925e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop customerID if exists\n",
    "if 'customerID' in df.columns:\n",
    "    df = df.drop('customerID', axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "826859ca-e260-459b-b72c-c6c388146e1c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "CustomerID               0\n",
       "Age                      0\n",
       "Gender                   0\n",
       "Tenure                   0\n",
       "Usage Frequency          0\n",
       "Support Calls            0\n",
       "Payment Delay            0\n",
       "Subscription Type        0\n",
       "Contract Length          0\n",
       "Total Spend              0\n",
       "Last Interaction         0\n",
       "Churn                64374\n",
       "dtype: int64"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "82e49cd5-12c7-4084-b43a-303d72cb88d4",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "a7135768-eee6-4547-a111-72f51676e67a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "✓ Loaded 64,374 rows × 12 columns\n",
      "  Churn rate : 47.37%\n",
      "  Missing    : 0\n",
      "\n"
     ]
    }
   ],
   "source": [
    "print(f\"\\n✓ Loaded {len(df):,} rows × {df.shape[1]} columns\")\n",
    "print(f\"  Churn rate : {df['Churn'].mean():.2%}\")\n",
    "print(f\"  Missing    : {df.isnull().sum().sum()}\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ffd5c5a6-a215-4e53-9cbf-48a56c98c23e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "============================================================\n",
      "CUSTOMER CHURN ANALYSIS\n",
      "============================================================\n"
     ]
    }
   ],
   "source": [
    "print(\"=\" * 60)\n",
    "print(\"CUSTOMER CHURN ANALYSIS\")\n",
    "print(\"=\" * 60)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "0a6f22a6-2d3c-41df-957b-562be7c14661",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "NUM_COLS = [\"Age\", \"Tenure\", \"Usage Frequency\", \"Support Calls\",\n",
    "            \"Payment Delay\", \"Total Spend\", \"Last Interaction\"]\n",
    "CAT_COLS = [\"Gender\", \"Subscription Type\", \"Contract Length\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "9a2d48d2-cb60-4570-8519-53cba23517db",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── 2. EDA Plots ──────────────────────────────────────────────────────────────\n",
    "def save(name: str) -> None:\n",
    "    path = os.path.join(OUTPUT_DIR, name)\n",
    "    plt.savefig(path, bbox_inches=\"tight\")\n",
    "    plt.close()\n",
    "    print(f\"  Saved → {path}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "1391b03c-a7d9-4ef6-a942-2e99679d94e6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Saved → churn_plots\\01_churn_distribution.png\n"
     ]
    }
   ],
   "source": [
    "# 2a. Churn distribution\n",
    "fig, axes = plt.subplots(1, 2, figsize=(12, 4))\n",
    "churn_counts = df[\"Churn\"].value_counts()\n",
    "axes[0].pie(churn_counts, labels=[\"No Churn\", \"Churned\"],\n",
    "            autopct=\"%1.1f%%\", colors=[\"#4C72B0\", \"#DD8452\"], startangle=90)\n",
    "axes[0].set_title(\"Churn Distribution\", fontweight=\"bold\")\n",
    "sns.countplot(x=\"Churn\", data=df, ax=axes[1], palette=[\"#4C72B0\", \"#DD8452\"])\n",
    "axes[1].set_xticklabels([\"No Churn (0)\", \"Churned (1)\"])\n",
    "axes[1].set_title(\"Churn Counts\", fontweight=\"bold\")\n",
    "for bar in axes[1].patches:\n",
    "    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 150,\n",
    "                 f\"{int(bar.get_height()):,}\", ha=\"center\", fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "save(\"01_churn_distribution.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "cdcc5d2a-a18a-4550-b80e-0af20d274185",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Saved → churn_plots\\02_numeric_distributions.png\n"
     ]
    }
   ],
   "source": [
    "# 2b. Numeric distributions by churn\n",
    "fig, axes = plt.subplots(2, 4, figsize=(18, 8))\n",
    "axes = axes.flatten()\n",
    "for i, col in enumerate(NUM_COLS):\n",
    "    for val, color, lbl in [(0, \"#4C72B0\", \"No Churn\"), (1, \"#DD8452\", \"Churned\")]:\n",
    "        axes[i].hist(df[df[\"Churn\"] == val][col], bins=30, alpha=0.6,\n",
    "                     color=color, label=lbl, density=True)\n",
    "    axes[i].set_title(col, fontweight=\"bold\")\n",
    "    axes[i].legend(fontsize=8)\n",
    "axes[-1].axis(\"off\")\n",
    "plt.suptitle(\"Feature Distributions by Churn\", fontsize=15, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "save(\"02_numeric_distributions.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "00b03e62-af58-435c-9792-e6ca1c7b8f88",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Saved → churn_plots\\03_categorical_churn_rates.png\n"
     ]
    }
   ],
   "source": [
    "# 2c. Categorical churn rates\n",
    "fig, axes = plt.subplots(1, 3, figsize=(16, 5))\n",
    "for i, col in enumerate(CAT_COLS):\n",
    "    rate = df.groupby(col)[\"Churn\"].mean().sort_values(ascending=False)\n",
    "    rate.plot(kind=\"bar\", ax=axes[i], color=\"#DD8452\", edgecolor=\"black\")\n",
    "    axes[i].set_title(f\"Churn Rate by {col}\", fontweight=\"bold\")\n",
    "    axes[i].set_ylabel(\"Churn Rate\")\n",
    "    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=30, ha=\"right\")\n",
    "    axes[i].set_ylim(0, 0.8)\n",
    "    for bar in axes[i].patches:\n",
    "        axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,\n",
    "                     f\"{bar.get_height():.1%}\", ha=\"center\", fontsize=9)\n",
    "plt.suptitle(\"Churn Rate by Categorical Features\", fontsize=14, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "save(\"03_categorical_churn_rates.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "b095857d-cb74-4275-b778-25c8cd35fc57",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Saved → churn_plots\\04_correlation_heatmap.png\n",
      "\n",
      "✓ EDA plots saved.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# 2d. Correlation heatmap\n",
    "plt.figure(figsize=(10, 7))\n",
    "corr = df[NUM_COLS + [\"Churn\"]].corr()\n",
    "mask = np.triu(np.ones_like(corr, dtype=bool))\n",
    "sns.heatmap(corr, mask=mask, annot=True, fmt=\".2f\",\n",
    "            cmap=\"coolwarm\", center=0, linewidths=0.5)\n",
    "plt.title(\"Correlation Matrix\", fontsize=14, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "save(\"04_correlation_heatmap.png\")\n",
    " \n",
    "print(\"\\n✓ EDA plots saved.\\n\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "aaca0dca-dceb-4ce1-a5d9-738a32997e2a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train : 51,499 | Test : 12,875\n"
     ]
    }
   ],
   "source": [
    "# ── 3. Preprocessing ──────────────────────────────────────────────────────────\n",
    "df_model = df.drop(columns=[\"CustomerID\"]).copy()\n",
    "le = LabelEncoder()\n",
    "for col in CAT_COLS:\n",
    "    df_model[col] = le.fit_transform(df_model[col])\n",
    " \n",
    "X = df_model.drop(columns=[\"Churn\"])\n",
    "y = df_model[\"Churn\"]\n",
    " \n",
    "X_train, X_test, y_train, y_test = train_test_split(\n",
    "    X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y\n",
    ")\n",
    " \n",
    "scaler = StandardScaler()\n",
    "X_train_sc = scaler.fit_transform(X_train)\n",
    "X_test_sc  = scaler.transform(X_test)\n",
    " \n",
    "print(f\"Train : {len(X_train):,} | Test : {len(X_test):,}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "f67ab791-344d-45e6-a7e0-bd1a8ce5f76a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "--- Logistic Regression ---\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    No Churn       0.84      0.83      0.83      6776\n",
      "     Churned       0.81      0.82      0.82      6099\n",
      "\n",
      "    accuracy                           0.83     12875\n",
      "   macro avg       0.83      0.83      0.83     12875\n",
      "weighted avg       0.83      0.83      0.83     12875\n",
      "\n",
      "ROC-AUC : 0.9020\n",
      "\n",
      "--- Random Forest ---\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "    No Churn       1.00      1.00      1.00      6776\n",
      "     Churned       1.00      1.00      1.00      6099\n",
      "\n",
      "    accuracy                           1.00     12875\n",
      "   macro avg       1.00      1.00      1.00     12875\n",
      "weighted avg       1.00      1.00      1.00     12875\n",
      "\n",
      "ROC-AUC : 1.0000\n"
     ]
    }
   ],
   "source": [
    "# ── 4. Models ─────────────────────────────────────────────────────────────────\n",
    "print(\"\\n--- Logistic Regression ---\")\n",
    "lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)\n",
    "lr.fit(X_train_sc, y_train)\n",
    "lr_pred  = lr.predict(X_test_sc)\n",
    "lr_proba = lr.predict_proba(X_test_sc)[:, 1]\n",
    "print(classification_report(y_test, lr_pred, target_names=[\"No Churn\", \"Churned\"]))\n",
    "lr_auc = roc_auc_score(y_test, lr_proba)\n",
    "print(f\"ROC-AUC : {lr_auc:.4f}\")\n",
    " \n",
    "print(\"\\n--- Random Forest ---\")\n",
    "rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)\n",
    "rf.fit(X_train, y_train)\n",
    "rf_pred  = rf.predict(X_test)\n",
    "rf_proba = rf.predict_proba(X_test)[:, 1]\n",
    "print(classification_report(y_test, rf_pred, target_names=[\"No Churn\", \"Churned\"]))\n",
    "rf_auc = roc_auc_score(y_test, rf_proba)\n",
    "print(f\"ROC-AUC : {rf_auc:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "6169800c-7ac9-4b35-8585-51c422060906",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "  Saved → churn_plots\\05_model_evaluation.png\n"
     ]
    }
   ],
   "source": [
    "# ── 5. Evaluation Plots ───────────────────────────────────────────────────────\n",
    "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
    " \n",
    "for ax, preds, title in [\n",
    "    (axes[0, 0], lr_pred, \"Logistic Regression\"),\n",
    "    (axes[0, 1], rf_pred, \"Random Forest\")\n",
    "]:\n",
    "    ConfusionMatrixDisplay(\n",
    "        confusion_matrix(y_test, preds),\n",
    "        display_labels=[\"No Churn\", \"Churned\"]\n",
    "    ).plot(ax=ax, cmap=\"Blues\", colorbar=False)\n",
    "    ax.set_title(f\"{title}\\nConfusion Matrix\", fontweight=\"bold\")\n",
    " \n",
    "for proba, label, color in [\n",
    "    (lr_proba, f\"LR  (AUC={lr_auc:.3f})\", \"#4C72B0\"),\n",
    "    (rf_proba, f\"RF  (AUC={rf_auc:.3f})\", \"#DD8452\")\n",
    "]:\n",
    "    fpr, tpr, _ = roc_curve(y_test, proba)\n",
    "    axes[1, 0].plot(fpr, tpr, label=label, color=color, lw=2)\n",
    "axes[1, 0].plot([0, 1], [0, 1], \"k--\", lw=1)\n",
    "axes[1, 0].set_xlabel(\"False Positive Rate\")\n",
    "axes[1, 0].set_ylabel(\"True Positive Rate\")\n",
    "axes[1, 0].set_title(\"ROC Curves\", fontweight=\"bold\")\n",
    "axes[1, 0].legend()\n",
    " \n",
    "feat_imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()\n",
    "feat_imp.plot(kind=\"barh\", ax=axes[1, 1], color=\"#4C72B0\")\n",
    "axes[1, 1].set_title(\"Feature Importance (Random Forest)\", fontweight=\"bold\")\n",
    "axes[1, 1].set_xlabel(\"Importance Score\")\n",
    " \n",
    "plt.suptitle(\"Model Evaluation Dashboard\", fontsize=16, fontweight=\"bold\")\n",
    "plt.tight_layout()\n",
    "save(\"05_model_evaluation.png\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "3b25f2e8-cf3b-456f-972b-f6137bc7e8f7",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "============================================================\n",
      "SUMMARY\n",
      "============================================================\n",
      "  Overall churn rate : 47.37%\n",
      "  Logistic Regression ROC-AUC : 0.9020\n",
      "  Random Forest       ROC-AUC : 1.0000\n",
      "  Top 3 predictors   : Payment Delay, Support Calls, Tenure\n",
      "\n",
      "  All plots saved to ./churn_plots/\n",
      "============================================================\n"
     ]
    }
   ],
   "source": [
    "# ── 6. Summary ────────────────────────────────────────────────────────────────\n",
    "print(\"\\n\" + \"=\" * 60)\n",
    "print(\"SUMMARY\")\n",
    "print(\"=\" * 60)\n",
    "print(f\"  Overall churn rate : {df['Churn'].mean():.2%}\")\n",
    "print(f\"  Logistic Regression ROC-AUC : {lr_auc:.4f}\")\n",
    "print(f\"  Random Forest       ROC-AUC : {rf_auc:.4f}\")\n",
    "top3 = pd.Series(rf.feature_importances_, index=X.columns).nlargest(3)\n",
    "print(f\"  Top 3 predictors   : {', '.join(top3.index.tolist())}\")\n",
    "print(f\"\\n  All plots saved to ./{OUTPUT_DIR}/\")\n",
    "print(\"=\" * 60)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.6"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
