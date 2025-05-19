Breast Cancer:-

Breast cancer is cancer that forms in the cells of the breasts.
It arises in the lining cells (epithelium) of the ducts (85%) or lobules (15%) in the glandular tissue of the breast.
Initially, the cancerous growth is confined to the duct or lobule (“in situ”) where it generally causes no symptoms and
has minimal potential for spread (metastasis).
Most types of breast cancer are easy to diagnose by microscopic analysis of a sample - or biopsy - of the affected
area of the breast.

 Also, there are types of breast cancer that require specialized lab exams.
This breast cancer prediction project objective is to identify the breast cancer from the datas of
patient's internal body features such as radius_mean ,perimeter_mean, area_mean, concavity_mean
concave points_mean, compactness_mean, radius_worst, perimeter_worst, area_worst, concave points_worst.

-The project that follows data preprocessing, modelling, accuracy score, and finally data analysis.
-As from the above data it previously detect breast cancer prediction. So this project mainly avoids the breast cancer.

Clean Workflow:
    Reads the dataset, drops unnecessary columns, handles missing values.
    Label encodes the target variable properly.

Data Preprocessing:
    Standard scaling of features before feeding into logistic regression (very important!).
    One-hot encoding is applied (though not necessary after label encoding — more on that below).

Model Building:
    Trains a Logistic Regression classifier.
    Uses train/test split with a fixed random_state.

Model Evaluation:
    Uses confusion matrix, classification report, and accuracy — good variety of metrics.

EDA (Exploratory Data Analysis):
    Count plot of diagnosis.
    Correlation heatmap (great for understanding feature relationships).

Clustering in breast cancer prediction:-

1. Understanding Natural Groupings
    Without using the diagnosis label, clustering can reveal if benign and malignant samples naturally form separate groups.
    You can compare clusters with actual diagnosis to check how well they align.
 2. Dimensionality Reduction + Visualization
    With PCA (Principal Component Analysis), you can reduce 30+ features to 2D and then apply clustering for visual exploration.
 3. Anomaly Detection
    Clustering might help spot outliers — samples that don't fit typical benign or malignant profiles.

 This clustering mainly explore natural groupings in the data. The resulting clusters aligned closely with diagnosis labels, showing strong feature separability.


