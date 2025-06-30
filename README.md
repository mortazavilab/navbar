# navbar
This is a streamlit-based h5ad explorer to support some of our lab data. It's in early development, but will hopefully be useful - at least internally. It works best on smaller adatas with 20,000 or less cells but should be ok with adatas up to ~100k cells.

There will be more documentation as it become more robust.

Download all of the code, install streamlit, scanpy and all other packages. Then launch it through navbar.py, e.g.:
```
python code/navbar/navbar.py --server.maxUploadSize 8000 --browser.serverAddress 127.0.0.1 -- --h5ad ~/IGVF/IGVFFI5345SNRS_processed.h5ad
```
