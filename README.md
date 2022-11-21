# pydacc

## Pydacc will cluster your data using machine learning 🚀

Upload a CSV, it'll clean then sort your data into clusters and send it back with a new column that has cluster labels.


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

---

# The API

Demo and docs: https://pydacc-production.up.railway.app

Note: demo is using railway's free plan so might not always be available

## `clustering`

Clusters your data based on how many groups you want. This is set by the `k` parameter.

## `auto-clustering`
Same as **`clustering`** but the number of groups are automatically decided.

## Examples

> **Note:** There is a bug with how Swagger (docs) creates the request so trying it in the docs may not work :(
>
> Links to relevant issues:
> * [Using "parameter: List[str] = Form(...)" produces unusable Swagger #1700](https://github.com/tiangolo/fastapi/issues/1700)
> * [form, swagger and Lists #2865](https://github.com/tiangolo/fastapi/issues/2865)
> * [set type hint Optional[\List[\str]], but receive value as [\'a,b,c'], not [\'a', 'b', 'c'] #2960](https://github.com/tiangolo/fastapi/issues/2960)


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

---

# How to run

## Develop locally

Clone:
```bash
git clone https://github.com/batmanscode/pydacc.git
```
Then run the server with:
```bash
uvicorn api:app --reload
```

And optionally ` --host 0.0.0.0 --port 8080`.

**Addtional info**
* `pydacc/` contains the python library.
* `api.py` is a FastAPI wrapper that makes the functions in `pydacc/` available as an API.
* See [FastAPI](https://fastapi.tiangolo.com/#run-it) docs for more info about the server.

## Deploy as a service

[Railway](https://railway.app) has a good free tier. I've created a template for this project; you can create an account and deploy this yourself by using the button below!

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/vVJFwO?referralCode=Qtw9qU)
