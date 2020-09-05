import pandas as pd



def id_checker(df):
    """
    The identifier checker

    Parameters
    ----------
    df : dataframe
    
    Returns
    ----------
    The dataframe of identifiers
    """
    
    # Get the identifiers
    df_id = df[[var for var in df.columns 
        if df[var].nunique(dropna=True) == df[var].notnull().sum()]]
                
    return df_id


def nan_checker(df):
    """
    The NaN checker

    Parameters
    ----------
    df : dataframe

    Returns
    ----------
    The dataframe of variables with NaN, their proportion of NaN and dtype
    """

    # Get the variables with NaN, their proportion of NaN and dtype
    df_nan = pd.DataFrame([[var, df[var].isna().sum() / df.shape[0], df[var].dtype]
                            for var in df.columns if df[var].isna().sum() > 0],
                            columns=['var', 'proportion', 'dtype'])

    # Sort df_nan in accending order of the proportion of NaN
    df_nan = df_nan.sort_values(by='proportion', ascending=False).reset_index(drop=True)

    return df_nan


def cat_var_checker(df):
    """
    The categorical variable checker

    Parameters
    ----------
    df: the dataframe

    Returns
    ----------
    The dataframe of categorical variables and their number of unique value
    """

    # Get the dataframe of categorical variables and their number of unique value
    df_cat = pd.DataFrame([[var, df[var].nunique(dropna=False)]
                            for var in df.columns if df[var].dtype == 'object'],
                            columns=['var', 'nunique'])

    # Sort df_cat in accending order of the number of unique value
    df_cat = df_cat.sort_values(by='nunique', ascending=False).reset_index(drop=True)

    return df_cat