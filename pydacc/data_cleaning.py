def clean_data(
    path_to_csv,
    drop_columns=None,
    column_drop_threshold=0.99,
):
    """
    Reads a csv file into a Pandas Dataframe, removes empty columns and then drops NaN values in the remaining columns.

    Afterwards, it uses `klib` to reduce the memory of what's left. For example, int64 might be changed to int32/int16.

    Example
    -------
    >>> data = clean_data('raw_data.csv', drop_threshold=0.9)

    path_to_csv: str
      Path to a '.csv` file. Alternatively, can be passed in as a string

    drop_columns: str, list, default=None
      Columns that aren't needed.

    column_drop_threshold: float, default=0.99
      Drop column(s) if it has this proportion of NaN values..
      The default 0.99 will drop every column which has 99% or more missing values.

    returns: Pandas Dataframe object
    """

    import pandas as pd
    import klib
    from io import StringIO

    try:
        df = pd.read_csv(path_to_csv)
    except:
        df = pd.read_csv(StringIO(path_to_csv), sep =",")
    else:
        raise ValueError("data passed must be in a CSV format. Either as a '.csv' file or a string")

    # drop named columns
    if drop_columns:
        df.drop(columns=drop_columns, inplace=True)

    # drop columns with a missing values threshold
    if column_drop_threshold:
        df.dropna(
            axis="columns",
            thresh=(1 - column_drop_threshold) * df.shape[0],
            inplace=True,
        )

    # drop NaN values from remaining data
    df.dropna(axis="index", inplace=True)

    # since data is already cleaned, this is to reduce the memory
    # by changing bit precision like float64 to float32 or float16
    klib.data_cleaning(df)

    # get a list of columns left after cleaning
    features = []
    for feature in df.columns:
        features.append(feature)

    return df, features


def get_common_items(list1, list2):
    """
    Compares two lists and returns the common items in a new list. Used internally.

    Example
    -------
    >>> list1 = [1, 2, 3, 4]
    >>> list2 = [2, 3, 4, 5]
    >>> common = get_common_items(list1, list2)

    list1: list

    list2: list

    returns: list
    """
    common = [value for value in list1 if value in list2]
    return common
