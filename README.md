# Navbar

**Navbar** is a Streamlit-based explorer for `.h5ad` single-cell datasets for our IGVF projects, designed for interactive visualization and QC of our lab data. It is under active development and is most responsive with datasets of 20,000 cells or fewer, but can handle datasets up to ~100,000 cells. It is hopefully useful to others, especially if you use some of the same adata obs fields that we use.

---

## Features

- **Splash Dataset Selector:** Choose from a list of datasets (local files or URLs) via a CSV config file before loading.
- **Flexible Data Loading:** Load `.h5ad` files from local paths, remote URLs, or by uploading directly in the sidebar.
- **Tabbed Interface:**  
  - **Summary Info:** Overview of dataset, metadata, and available slots.
  - **QC:** Plate heatmaps, knee plots, and violin plots for QC metrics.
  - **Embedding Plot:** Visualize UMAP, t-SNE, or PCA embeddings colored by metadata.
  - **Gene Expression:** Plot gene expression on embeddings and as plate heatmaps.
  - **Marker Genes:** Calculate and view marker genes for clusters or groups.
  - **Pathway Analysis:** GSEA and other enrichment analyses.
  - **Pseudobulk PCA/DEG:** Pseudobulk PCA and differential expression.
- **Progress Bar:** Visual feedback when downloading large remote files.
- **Downloadable Plots:** Export figures as PNG or SVG.
- **Session State:** Remembers your selections and allows switching datasets without restarting the app.

---

## Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/yourlab/navbar.git
    cd navbar
    ```

2. **Install dependencies:**

- [streamlit](https://streamlit.io/)
- [scanpy](https://scanpy.readthedocs.io/)
- [anndata](https://anndata.readthedocs.io/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [requests](https://docs.python-requests.org/)
- [scipy](https://scipy.org/)
- [gseapy](https://gseapy.readthedocs.io/) (for pathway analysis)
- [pydeseq2](https://pydeseq2.readthedocs.io/) (for pseudobulk DEG)


Install all core dependencies with:
```sh
pip install streamlit scanpy anndata matplotlib seaborn pandas numpy requests scipy gseapy pydeseq2
```
---

## Usage

### **Basic Usage**

To launch Navbar with a single `.h5ad` file:
```sh
python code/navbar/navbar.py -- --h5ad /path/to/your_data.h5ad
```
Or with a remote file:
```sh
python code/navbar/navbar.py -- --h5ad https://yourserver.org/data/mydata.h5ad
```

### **Using a CSV Config File**

You can provide a CSV file listing multiple datasets (local or remote) for splash selection.  
Each line should be:  
`dataID,dataPath`  
where `dataPath` is a local file path or a URL.

**Example `IGVF.csv`:**
```
Adrenal_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI1398CMEX/@@download/IGVFFI1398CMEX.h5ad
CX_HC_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI0968TUZT/@@download/IGVFFI0968TUZT.h5ad
DE_Pit_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI8973OQHG/@@download/IGVFFI8973OQHG.h5ad
Gastroc_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI7132PQNS/@@download/IGVFFI7132PQNS.h5ad
Gonads_Female_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI3402EZDO/@@download/IGVFFI3402EZDO.h5ad
Gonads_Male_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI6602LGHO/@@download/IGVFFI6602LGHO.h5ad
Heart_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI6644FMFS/@@download/IGVFFI6644FMFS.h5ad
Kidney_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI3941URLH/@@download/IGVFFI3941URLH.h5ad
Liver_Founder_8Cube,https://api.data.igvf.org/matrix-files/IGVFFI8026GCPM/@@download/IGVFFI8026GCPM.h5ad
bridge_supool,https://api.data.igvf.org/matrix-files/IGVFFI3320ZCCE/@@download/IGVFFI3320ZCCE.h5ad
```

**To launch with a config file:**
```sh
python code/navbar/navbar.py -- --config code/navbar/IGVF.csv
```

You will be presented with a splash page to select which dataset to load.

---

### **Other Options**

- **Upload a file:** Use the sidebar in the app to upload a `.h5ad` file directly.
- **Enter a URL:** Paste a remote `.h5ad` URL in the sidebar to load it.
- **Enter a file path:** Type a local file path in the sidebar.

---

## Notes

- For best performance, use datasets with ≤20,000 cells for interactive exploration.
- Some features (e.g., GSEA, pseudobulk DEG) require additional dependencies.
- The app displays its version number in the sidebar.

---

## Example Command Lines

```sh
# Launch with a single local file
python code/navbar/navbar.py -- --h5ad /local/path/IGVF/Heart_integrated_processed_annotated_fixed_subsampled20k.h5ad

# Launch with a remote file
python code/navbar/navbar.py -- --h5ad https://api.data.igvf.org/matrix-files/IGVFFI3320ZCCE/@@download/IGVFFI3320ZCCE.h5ad

# Launch with a config CSV for splash selection
python code/navbar/navbar.py -- --config IGVF.csv
```

---

## License

MIT License

Copyright (c) Ali Mortazavi

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
