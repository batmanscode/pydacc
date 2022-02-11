import shutil
from typing import List, Optional
import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile, Form, BackgroundTasks
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import Field
from starlette.background import BackgroundTask
import os
from datetime import date
from pydacc.clustering import train_clustering_model, automl_clustering, predict_cluster_label , assign_cluster_labels


request_example = """
### How to make a request to the API and download the output CSV:
```
import requests
import pandas as pd
import io


url = 'https://...'
input_csv = 'example.csv' # in the repo
endpoint = 'auto-clustering'
output_file_name = 'cluster_model'
output_file_path = None

files = {
    'path_to_csv': (os.path.basename(input_csv), open(input_csv, 'rb'), 'text/csv'),
    'column_drop_threshold': (None, '0.99'),
    'file_name': (None, output_file_name),
    'drop_columns': (None, 'id,category,account_code'),
    'categorical_columns': (None, 'city'),
    'numerical_columns': (None, 'TotalOrderCount,TotalOrderValue,outstanding_debt,TotalReturnedValue,TotalReturnedQty,TotalStock'),
    'output_format': (None, 'csv'),
}

response = requests.post(f'{url}/{endpoint}/', files=files)


df = pd.read_csv(io.StringIO(response.text))

if output_file_path:
    df.to_csv(f'{output_file_path}/{output_file_name}.csv', index=False)
else:
    df.to_csv(f'{output_file_name}.csv', index=False)
```
"""

app = FastAPI(
    title="AutoML 🤖: Clustering", description=request_example, version="1.0.0"
)


def save_to_disk(uploaded_file):
    """
    Saves the uploaded file to disk since for whatever reason FileUlpload only works properly in `clustering`.
    """
    with open(uploaded_file.filename, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)


def delete_temp(
    path: str = str(os.getcwd()), file_extensions: List[str] = [".pkl", ".csv", ".json"]
):
    """
    Background task to delete created files after it's sent to the user.

    Note: `os.getcwd()` is used since the created and uploaded files are in the cwd
    """
    for f in os.listdir(path):
        for extension in file_extensions:
            if not f.endswith(extension):
                continue
            os.remove(os.path.join(path, f))


@app.get("/", response_class=PlainTextResponse)
async def running():
    note = """
The API is running cheif! 🙌🏻

Note: add "/docs" to the url to get the swagger ui docs or "/redoc"
  """
    return note


#### docs ####

# for each parameter
# having them here since they share most of it
path_to_csv_doc = "Path to a `.csv` file. Alternatively, can be sent as a string"

k_doc = "number of clusters to create"

column_drop_threshold_doc = """
Drop column(s) if it has this proportion of NaN values.
The default 0.99 will drop every column which has 99% or more missing values.
"""

file_name_doc = "Name of the saved model file. Current date is added to the end of the name."

drop_columns_doc = "Columns that aren't needed. Set to None if you want to keep these."

categorical_columns_doc = "Specify columns with catergorical (non-numerical) data"

numerical_columns_doc = "Specify columns with numerical data"

ignore_features_doc = """Things that aren't useful to clustering but can be otherwise useful.
A unique id column for example would make searching clusters easier but wouldn't be useful for creating the clusters.
"""

output_format_doc = "'csv' or 'json'"

csv_string_doc = "CSV as a string. Requires tripple quotes to preserve line breaks"

# functions
clustering_doc = """Reads a csv file into a Pandas Dataframe obeject and creates a kmeans clustering model. Calls `data_cleaning` and `get_common_items` internally.

Except for `path_to_csv` and `k`, all other params are optional.

**Returns:** a CSV with labels added to the input data after creating a model
"""

auto_clustering_doc = """Automatically finds the best k by iterating through values of k and selecting the one with the best silhouette coefficient. Not always the best but usually good enough.

Calls the `train_clustering_model` internally.

**Returns:** a CSV with labels added to the input data after creating a model
"""

clustering_csv_string_doc = """Reads a csv string into a Pandas Dataframe obeject and creates a kmeans clustering model. Calls `data_cleaning` and `get_common_items` internally.

Except for `csv_string` and `k`, all other params are optional.

**Returns:** a CSV with labels added to the input data after creating a model
"""


@app.post("/clustering", description=clustering_doc)
def clustering(
    background_tasks: BackgroundTasks,
    path_to_csv: UploadFile = File(..., description=path_to_csv_doc),
    k: int = Form(..., description=k_doc),
    column_drop_threshold: float = Form(0.99, description=column_drop_threshold_doc),
    file_name: str = Form("cluster_model", description=file_name_doc),
    drop_columns: List[str] = Form(None, description=drop_columns_doc),
    categorical_columns: List[str] = Form(None, description=categorical_columns_doc),
    numerical_columns: List[str] = Form(None, description=numerical_columns_doc),
    ignore_features: List[str] = Form(None, description=ignore_features_doc),
    output_format: str = Form("csv", description=output_format_doc),
):

    save_to_disk(path_to_csv)

    # trained model
    train_clustering_model(
        path_to_csv=path_to_csv.filename,
        k=k,
        column_drop_threshold=column_drop_threshold,
        save_model=True,  # the api can't return a variable, it has to be a saved file
        file_name=file_name,
        drop_columns=drop_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        ignore_features=ignore_features,
    )

    # assign labels
    assign_cluster_labels(
        saved_model=f"{file_name}_{date.today()}",
        save_output=True,
        file_name=file_name,
        output_format=output_format,
    )

    if file_name is None:
        file_name = saved_model

    if output_format == "json":
        cluster_labels = f"{file_name}.json"
    if output_format == "csv":
        cluster_labels = f"{file_name}.csv"

    return FileResponse(cluster_labels, background=BackgroundTask(delete_temp))


@app.post("/auto-clustering", description=auto_clustering_doc)
def auto_clustering(
    background_tasks: BackgroundTasks,
    path_to_csv: UploadFile = File(..., description=path_to_csv_doc),
    column_drop_threshold: float = Form(0.99, description=column_drop_threshold_doc),
    file_name: str = Form("cluster_model", description=file_name_doc),
    drop_columns: List[str] = Form(None, description=drop_columns_doc),
    categorical_columns: List[str] = Form(None, description=categorical_columns_doc),
    numerical_columns: List[str] = Form(None, description=numerical_columns_doc),
    ignore_features: List[str] = Form(None, description=ignore_features_doc),
    output_format: str = Form("csv", description=output_format_doc),
):

    save_to_disk(path_to_csv)

    automl_clustering(
        path_to_csv=path_to_csv.filename,
        column_drop_threshold=column_drop_threshold,
        save_model=True,
        file_name=file_name,
        drop_columns=drop_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        ignore_features=ignore_features,
    )

    # assign labels
    assign_cluster_labels(
        saved_model=f"{file_name}_{date.today()}",
        save_output=True,
        file_name=file_name,
        output_format=output_format,
    )

    if file_name is None:
        file_name = saved_model

    if output_format == "json":
        cluster_labels = f"{file_name}.json"
    if output_format == "csv":
        cluster_labels = f"{file_name}.csv"

    return FileResponse(cluster_labels, background=BackgroundTask(delete_temp))


# end points that can take strings instead of files
@app.post("/clustering-csv-string", summary="Same as `clustering` but takes the CSV as a string instead of a file", description=clustering_csv_string_doc)
def clustering_csv_string(
    background_tasks: BackgroundTasks,
    csv_string: str = Field(..., description=csv_string_doc),
    k: int = Field(..., description=k_doc),
    column_drop_threshold: float = Field(0.99, description=column_drop_threshold_doc),
    file_name: str = Field("cluster_model", description=file_name_doc),
    drop_columns: Optional[List[str]] = Field(None, description=drop_columns_doc),
    categorical_columns: Optional[List[str]] = Field(None, description=categorical_columns_doc),
    numerical_columns: Optional[List[str]] = Field(None, description=numerical_columns_doc),
    ignore_features: Optional[List[str]] = Field(None, description=ignore_features_doc),
    output_format: str = Field("csv", description=output_format_doc),
):

    # trained model
    train_clustering_model(
        path_to_csv=csv_string,
        k=k,
        column_drop_threshold=column_drop_threshold,
        save_model=True,  # the api can't return a variable, it has to be a saved file
        file_name=file_name,
        drop_columns=drop_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        ignore_features=ignore_features,
    )

    # assign labels
    assign_cluster_labels(
        saved_model=f"{file_name}_{date.today()}",
        save_output=True,
        file_name=file_name,
        output_format=output_format,
    )

    if file_name is None:
        file_name = saved_model

    if output_format == "json":
        cluster_labels = f"{file_name}.json"
    if output_format == "csv":
        cluster_labels = f"{file_name}.csv"

    return FileResponse(cluster_labels, background=BackgroundTask(delete_temp))


@app.post("/auto-clustering-csv-string", summary="Same as `auto-clustering` but takes the CSV as a string instead of a file", description=auto_clustering_doc)
def auto_clustering_csv_string(
    background_tasks: BackgroundTasks,
    csv_string: str = Field(..., description=csv_string_doc),
    column_drop_threshold: float = Field(0.99, description=column_drop_threshold_doc),
    file_name: str = Field("cluster_model", description=file_name_doc),
    drop_columns: Optional[List[str]] = Field(None, description=drop_columns_doc),
    categorical_columns: Optional[List[str]] = Field(None, description=categorical_columns_doc),
    numerical_columns: Optional[List[str]] = Field(None, description=numerical_columns_doc),
    ignore_features: Optional[List[str]] = Field(None, description=ignore_features_doc),
    output_format: str = Field("csv", description=output_format_doc),
):

    automl_clustering(
        path_to_csv=csv_string,
        column_drop_threshold=column_drop_threshold,
        save_model=True,
        file_name=file_name,
        drop_columns=drop_columns,
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        ignore_features=ignore_features,
    )

    # assign labels
    assign_cluster_labels(
        saved_model=f"{file_name}_{date.today()}",
        save_output=True,
        file_name=file_name,
        output_format=output_format,
    )

    if file_name is None:
        file_name = saved_model

    if output_format == "json":
        cluster_labels = f"{file_name}.json"
    if output_format == "csv":
        cluster_labels = f"{file_name}.csv"

    return FileResponse(cluster_labels, background=BackgroundTask(delete_temp))
