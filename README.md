# pydacc
### Data cleaning and clustering API

Demo: https://pydacc-production.up.railway.app

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template/vVJFwO?referralCode=Qtw9qU)

To run locally:
```bash
git clone https://github.com/batmanscode/pydacc.git
```
Then
```bash
uvicorn api:app
```

And optionally ` --host 0.0.0.0 --port 8080`

For development, using `--reload` will make things a little easier.

---

**Addtional info**
* `pydacc/` contains the python library.
* `api.py` is a FastAPI wrapper that makes the functions in `pydacc/` available as an API.
