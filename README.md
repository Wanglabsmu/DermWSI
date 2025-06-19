# DermWSI - Dermatopathology Whole Slide Image Analysis Pipeline

This project provides a complete pathology slide analysis pipeline, including image preprocessing, feature extraction, and prediction analysis.

## Quick start

System Requirement: Linux (tested on CentOS 7.9).

### Step1. Download pipeline

```bash
git clone https://github.com/Wanglabsmu/DermWSI
cd DermWSI
```

### Step2. Creat  and activate environment
```bash
## creat environment
conda env create -f environment.yaml

## activate environment
conda activate DermWSI
```

### Step3. Upload WSI

Upload WSI to DATA/WSI, e.g. ./DATA/WSI/case.ndpi

### Step4. Analysis

```
sh main.sh ./DATA/WSI/case.ndpi
```



## Outputs

(1) The prediction label;

(2) The key patches of the top 10% contributive feature regions and an annotated WSI marking these key regions are stored in ./DATA/OUTPUT/case
