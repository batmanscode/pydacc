import shutil
from typing import List, Optional, Union
import pandas as pd
from fastapi import FastAPI, File, Query, UploadFile, Form, BackgroundTasks, Body
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask
import os
from datetime import date
from pydacc.clustering import train_clustering_model, automl_clustering, predict_cluster_label , assign_cluster_labels

from fastapi.middleware.cors import CORSMiddleware # temp for cors


description = """
**Pydacc will cluster your data using machine learning 🚀.**

Upload a CSV, it\'ll clean then sort your data into clusters and send it back with a new column that has cluster labels.


For example, this:
>
>Item | Type
>---|---
>Apples | Fruit
>Bananas | Fruit
>Cat | Fren
>Kodiak bear | Fren
>
Will get sent back to you as this:
>
>Item | Type |Cluster
>---|---|---
>Apples | Fruit | Cluster 0
>Bananas | Fruit | Cluster 0
>Cat | Fren | Cluster 1
>Kodiak bear | Fren | Cluster 1


## Clustering
Clusters your data based on how many groups you want. This is set by the `k` parameter.

## Auto Clustering
Same as **Clustering** but the number of groups are automatically decided.

## Examples

> **Note:** There is a bug with how Swagger (docs) creates the request so trying it in the docs may not work :(
>
> Links to relevant issues:
> * [Using "parameter: List[str] = Form(...)" produces unusable Swagger #1700](https://github.com/tiangolo/fastapi/issues/1700)
> * [form, swagger and Lists #2865](https://github.com/tiangolo/fastapi/issues/2865)
> * [set type hint Optional\[List\[str\]\], but receive value as \['a,b,c'\], not \['a', 'b', 'c'\] #2960](https://github.com/tiangolo/fastapi/issues/2960)


### Python:
```python
import requests
import pandas as pd
import io
import os

url = 'https://pydacc-production.up.railway.app'
input_csv = 'example.csv' # in the repo
endpoint = 'auto-clustering'
output_file_name = 'cluster_model'
output_file_path = None # save csv to custom path

files = [
    ('path_to_csv', (os.path.basename(input_csv), open(input_csv, 'rb'), 'text/csv')),
    ('column_drop_threshold', (None, '0.99')),
    ('file_name', (None, 'cluster_model')),
    ('drop_columns', (None, 'category')),
    ('drop_columns', (None, 'account_code')),
    ('categorical_columns', (None, 'city')),
    ('numerical_columns', (None, 'TotalOrderCount')),
    ('numerical_columns', (None, 'TotalOrderValue')),
    ('numerical_columns', (None, 'outstanding_debt')),
    ('numerical_columns', (None, 'TotalReturnedValue')),
    ('numerical_columns', (None, 'TotalReturnedQty')),
    ('numerical_columns', (None, 'TotalStock')),
    ('ignore_features', (None, 'id')),
    ('ignore_features', (None, 'name')),
    ('output_format', (None, 'csv')),
]

response = requests.post(f'{url}/{endpoint}', files=files)

# create dataframe
df = pd.read_csv(io.StringIO(response.text))

# save file
if output_file_path:
    df.to_csv(f'{output_file_path}/{output_file_name}.csv', index=False)
else:
    df.to_csv(f'{output_file_name}.csv', index=False)

# print dataframe
df
```

### Curl:
```curl
curl -X 'POST' \  
  'https://pydacc-production.up.railway.app/auto-clustering' \  
  -H 'accept: application/json' \  
  -H 'Content-Type: multipart/form-data' \  
  -F 'path_to_csv=@example.csv;type=text/csv' \  
  -F 'column_drop_threshold=0.99' \  
  -F 'file_name=cluster_model' \  
  -F 'drop_columns=category' \  
  -F 'drop_columns=account_code' \  
  -F 'categorical_columns=city' \  
  -F 'numerical_columns=TotalOrderCount' \  
  -F 'numerical_columns=TotalOrderValue' \  
  -F 'numerical_columns=outstanding_debt' \  
  -F 'numerical_columns=TotalReturnedValue' \  
  -F 'numerical_columns=TotalReturnedQty' \  
  -F 'numerical_columns=TotalStock' \  
  -F 'ignore_features=id' \  
  -F 'ignore_features=name' \  
  -F 'output_format=csv'
```
"""

app = FastAPI(
    title="Pydacc: Data Cleaning and Clustering", description=description, version="1.0.0"
)

# TEMP FOR CORS
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

drop_columns_doc = "Columns you want to drop; these will be removed from the clustered data sent back to you. Set to None if you want to keep these."

categorical_columns_doc = "Specify columns with catergorical (non-numerical) data"

numerical_columns_doc = "Specify columns with numerical data"

ignore_features_doc = """Things that aren't useful to clustering that you want ignored when training but not removed from the output file.
A unique id column for example would make searching clusters easier but wouldn't be useful for creating the clusters.
"""

output_format_doc = "'csv' or 'json'"

# functions
clustering_doc = """Reads a csv file into a Pandas Dataframe obeject and creates a kmeans clustering model. Calls `data_cleaning` and `get_common_items` internally.

Except for `path_to_csv` and `k`, all other params are optional.

**Returns:** a CSV with labels added to the input data after creating a model
"""

auto_clustering_doc = """Automatically finds the best k by iterating through values of k and selecting the one with the best silhouette coefficient. Not always the best but usually good enough.

Calls the `train_clustering_model` internally.

**Returns:** a CSV with labels added to the input data after creating a model
"""


@app.post("/clustering", description=clustering_doc)
def clustering(
    background_tasks: BackgroundTasks,
    path_to_csv: UploadFile = File(..., description=path_to_csv_doc),
    k: int = Form(..., description=k_doc),
    column_drop_threshold: float = Form(0.99, description=column_drop_threshold_doc),
    file_name: str = Form("cluster_model", description=file_name_doc),
    drop_columns: Union[List[str], None] = Form(None, description=drop_columns_doc),
    categorical_columns: Union[List[str], None] = Form(None, description=categorical_columns_doc),
    numerical_columns: Union[List[str], None] = Form(None, description=numerical_columns_doc),
    ignore_features: Union[List[str], None] = Form(None, description=ignore_features_doc),
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
    drop_columns: Union[List[str], None] = Form(None, description=drop_columns_doc),
    categorical_columns: Union[List[str], None] = Form(None, description=categorical_columns_doc),
    numerical_columns: Union[List[str], None] = Form(None, description=numerical_columns_doc),
    ignore_features: Union[List[str], None] = Form(None, description=ignore_features_doc),
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
