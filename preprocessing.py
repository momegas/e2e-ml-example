from sklearn import preprocessing
import pandas as pd
import numpy as np


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Process data for inference."""
    # Create a copy of the dataframe
    data = df.copy()

    # Convert binary categorical variables to numeric
    label_encoder = preprocessing.LabelEncoder()
    label_columns = ["Gender", "Married", "Education", "Self_Employed"]

    for label_column in label_columns:
        data[label_column] = label_encoder.fit_transform(data[label_column])

    # Convert categorical variables to numeric
    categorical_columns = ["Dependents", "Property_Area"]

    dummies = pd.get_dummies(
        data[["Dependents"]],
        columns=["Dependents"],
        prefix=["Dependents_"],
        drop_first=True,
        dummy_na=True,
    )
    data = data.join(dummies)
    print(dummies.head())

    data = data.drop(["Property_Area", "Dependents"], axis=1)
    data = data.astype(np.float64)

    return data
