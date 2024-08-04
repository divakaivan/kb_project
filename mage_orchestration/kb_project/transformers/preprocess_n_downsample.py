import pandas as pd
from sklearn.utils import resample

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@transformer
def transform(data, *args, **kwargs):
    
    credit_card_nodes, merchant_nodes, transactions = data
    df = transactions.join(credit_card_nodes.set_index('cc_num'), on='cc_num').join(merchant_nodes.set_index('merchant'), on='merchant')
    
    # create new features
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['trans_hour'] = df['trans_date_trans_time'].dt.hour
    df['trans_dow'] = df['trans_date_trans_time'].dt.dayofweek
    
    numerical_features = ['amt'] 
    categorical_features = [
        'category', 'state',
        'job', 'trans_hour', 'trans_dow'
        ]
    target = 'is_fraud'  
    
    df = df[numerical_features + categorical_features + [target]]
    
    for cat in categorical_features:
        df = pd.concat([df, pd.get_dummies(df[cat], prefix=cat, drop_first=True)], axis=1)
        df.drop(cat, axis=1, inplace=True)
    
    # downsample majority
    df_not_fraud = df[df['is_fraud'] == 0].sample(frac=0.2, random_state=999)
    df_fraud = df[df['is_fraud'] == 1]
    df_undersampled = pd.concat([df_not_fraud, df_fraud])

    non_fraud_rows = df_undersampled[df_undersampled['is_fraud'] == 0].copy()
    fraud_rows = df_undersampled[df_undersampled['is_fraud'] == 1].copy()
    non_fraud_downsampled = resample(
            non_fraud_rows, 
            n_samples=len(fraud_rows), 
            replace=False, 
            random_state=42
        )
    balanced_df = pd.concat([fraud_rows, non_fraud_downsampled])
    balanced_df = balanced_df.reset_index(drop=True)

    return balanced_df


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'
