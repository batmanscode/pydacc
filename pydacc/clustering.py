from pydacc.data_cleaning import clean_data, get_common_items


#### training / create model ####


def train_clustering_model(
    path_to_csv,
    k,
    column_drop_threshold=0.99,
    save_model=True,
    file_name="cluster_model",
    drop_columns=None,
    categorical_columns=None,
    numerical_columns=None,
    ignore_features=None,
):
    """
     Reads a csv file into a Pandas Dataframe obeject and creates a kmeans clustering model. Calls `data_cleaning` and `get_common_items` internally.

     Except for `path_to_csv` and `k`, all other params are optional.

     Example
     -------
     >>> model = train_clustering_model('raw_data.csv', k=4)

     path_to_csv: str

     k: int
       number of clusters to create.

    column_drop_threshold: float, default=0.99
       Drop column(s) if it has this proportion of NaN values. Set to None if you want to keep these.
       The default 0.9 will drop every column which has 99% or more missing values.

    save_model: boolean, default=True
      Saves pickled model to local directory.

    file_name: str, default='cluster_model'
      Name of the saved model file. Current date is added to the end of the name. Ignored if `save_model=False`.

    drop_columns: str, list, default=None
      Columns that aren't needed. Set to None if you want to keep these.
      By default, it drops all unique values except for 'name' which is left to make searching possible.

    categorical_columns: list, default=None
      Specify columns with catergorical (non-numerical) data

    numerical_columns: list, default=None
      Specify columns with numerical data.

    ignore_features: list, default=None
      Things that aren't useful to clustering but can be otherwise useful.
      A unique id column for example would make searching clusters easier but wouldn't be useful for creating the clusters.

    returns: Trained model
    """

    # csv into pandas.Dataframe
    import pandas as pd

    # data = pd.read_csv(path_to_csv)
    data, features = clean_data(
        path_to_csv, column_drop_threshold=column_drop_threshold
    )

    # get_common_itemss() will use features left from data cleaning to filter out the dataframe
    # this way all the possible catergorical and numerical features can be passed as arguments
    # and the features left after cleaning will be compared to find what's catergorical/numberical among them

    # set up and create model
    from pycaret.clustering import setup, create_model, save_model, assign_model

    cluster_setup = setup(
        data,
        normalize=True,
        ignore_features=ignore_features,  # not useful for clustering, but useful for searching
        session_id=12,
        categorical_features=get_common_items(features, categorical_columns),
        numeric_features=get_common_items(features, numerical_columns),
        ignore_low_variance=True,
        pca=True,
        html=False,
        silent=True,
    )

    kmeans = create_model("kmeans", num_clusters=k)

    # save output

    if save_model:
        from datetime import date

        return save_model(kmeans, f"{file_name}_{date.today()}")
        print(f"Model saved to {file_name}_{date.today()}")
    else:
        return kmeans


def automl_clustering(
    path_to_csv,
    column_drop_threshold=0.99,
    save_model=True,
    file_name="cluster_model",
    drop_columns=None,
    categorical_columns=None,
    numerical_columns=None,
    ignore_features=None,
):

    """
    Automatically finds the best k by iterating through values of k and selecting the one with the best silhouette coefficient. Not always the best but usually good enough.

    Calls the `train_clustering_model` internally.

    Example
    -------
    >>> model = automl_clustering('raw_data.csv')

    path_to_csv: str

    column_drop_threshold: float, default=0.99
      Drop column(s) if it has this proportion of NaN values. Set to None if you want to keep these.
      The default 0.9 will drop every column which has 99% or more missing values.

    save_model: boolean, default=True
      Saves pickled model to local directory

    file_name: str, default='cluster_model'
      Name of the saved model file. Current date is added to the end of the name. Ignored if `save_model=False`

    drop_columns: str, list, default=None
      Columns that aren't needed. Set to None if you want to keep these.
      By default, it drops all unique values except for 'name' which is left to make searching possible

    categorical_columns: list, default=None
      Specify columns with catergorical (non-numerical) data

    numerical_columns: list, default=None
      Specify columns with numerical data

    ignore_features: list, default=None
      Things that aren't useful to clustering but can be otherwise useful.
      A unique id column for example would make searching clusters easier but wouldn't be useful for creating the clusters.

    returns: Trained model
    """
    from pycaret.clustering import get_config

    score = []  # save silhouette coefficients here
    k_space = [3, 4, 5, 6, 7, 8, 9, 10]  # search space

    for k in k_space:
        train_clustering_model(
            path_to_csv,
            k,
            column_drop_threshold=column_drop_threshold,
            save_model=save_model,
            file_name=file_name,
            drop_columns=drop_columns,
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            ignore_features=ignore_features,
        )

        silhouette = get_config("create_model_container")
        score.append(silhouette[0]["Silhouette"][0])

    # find the index of the highest silhouette coefficient
    index_best_k = score.index(max(score))

    # use that to find the best k value
    k = k_space[index_best_k]

    return train_clustering_model(
        path_to_csv,
        k,
        column_drop_threshold=column_drop_threshold,
        save_model=save_model,
        file_name=file_name,
        drop_columns=drop_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        ignore_features=ignore_features,
    )


#### add labels ####


def assign_cluster_labels(
    saved_model, save_output=True, file_name=None, output_format="csv"
):
    """
    Loads a saved model, assigns cluster labels to the input data and saves the dataframe as a csv or json file.

    Example
    -------
    >>> clusters = assign_cluster_labels('cluster_model')

    saved_model: str
      Path to saved model. Exclude the `.pkl` extension here.

    save_output: boolean, defaul=True
      Saves output

    file_name: str, default=None
      File name for the output. By default, it'll use the same name as 'saved_model`. Ignored if save_output=False.

    output_format: str, default='csv'
      Choose the output format to save the resulting dataframe as either a csv or json file. Ignored if save_output=False.

      returns: dataframe with cluster labels assigned to input data
    """

    # load saved model
    from pycaret.clustering import load_model

    saved_kmeans = load_model(saved_model)

    # assign labels
    from pycaret.clustering import assign_model

    kmeans_df = assign_model(saved_kmeans)

    if save_output:

        if file_name is None:
            file_name = saved_model

        if output_format == "csv":
            return kmeans_df.to_csv(f"{file_name}.csv", index=False)
        if output_format == "json":
            return kmeans_df.to_json(f"{file_name}.json")

    else:
        return kmeans_df


#### predict ####


def predict_cluster_label(
    saved_model, path_to_csv, save_output=True, file_name=None, output_format="csv"
):
    """
    Loads a saved model, predicticts on input data and saves the dataframe as a csv or json file.

    Important: The input data needs to have the same shape as the training data used to create the model.

    Example
    -------
    >>> predictions = predict_cluster_label(saved_model='cluste_model', path_to_csv='unseen_data.csv')

    saved_model: str
      Path to saved model. Exclude the `.pkl` extension here.

    path_to_csv: str
      Unseen data on which to make the prediction.

    save_output: boolean, defaul=True
      Saves output.

    file_name: str, default=None
      File name for the output. By default, it'll use the same name as 'saved_model`. Ignored if save_output=False.

    output_format: str, default='csv'
      Choose the output format to save the resulting dataframe as either a csv or json file. Ignored if save_output=False.

      returns: dataframe of predictions
    """

    import pandas as pd

    df = pd.read_csv(path_to_csv)

    # load saved model
    from pycaret.clustering import load_model

    saved_kmeans = load_model(saved_model)

    # make prediction
    from pycaret.clustering import predict_model

    kmeans_predictions = predict_model(model=saved_kmeans, data=df)

    if save_output:

        if file_name is None:
            file_name = saved_model

        if output_format == "csv":
            return kmeans_predictions.to_csv(f"{file_name}.csv", index=False)
        if output_format == "json":
            return kmeans_predictions.to_json(f"{file_name}.json")

    else:
        return kmeans_predictions
