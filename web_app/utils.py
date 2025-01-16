import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_categorical_columns(df: pd.DataFrame, encoding_type: str = 'label') -> pd.DataFrame:
    """
    Encodes categorical columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame with categorical columns.
        encoding_type (str): The encoding type, either 'label' or 'onehot'.

    Returns:
        pd.DataFrame: A DataFrame with encoded categorical columns.
    """
    df_encoded = df.copy()
    categorical_columns = df_encoded.select_dtypes(include=['category', 'object']).columns

    if encoding_type == 'label':
        for col in categorical_columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col])

    elif encoding_type == 'onehot':
        df_encoded = pd.get_dummies(df_encoded, columns=categorical_columns, drop_first=True)
    else:
        raise ValueError("Unsupported encoding_type. Use 'label' or 'onehot'.")
    
    return df_encoded

def encode_categorical_columns_training_encoder(df, label_encoders):
    # Use the loaded label_encoders to encode specified categorical columns
    encoded_columns = ['airline', 'origin_city', 'dest_city']
    for column in encoded_columns:
        if column in df.columns and column in label_encoders:
            df[column] = label_encoders[column].transform(df[column])
        else:
            raise ValueError(f"Label encoder for column {column} not found.")
    return df

        # example usage
        # come back to this later
        # encoded_input_df = encode_categorical_columns_training_encoder(input_df, label_encoder)


       # lets add the scaling here
       # encoded_input_scaled_df = scaler.transform(encoded_input_df)

# task 46
# Remember to label encode the data then scale ie the trimmed data then train for uniformity 

