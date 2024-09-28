# SOM-implementation-DA-2-Big-data-analytics

# Self-Organizing Maps for Fraud Detection

This project demonstrates how to use **Self-Organizing Maps (SOMs)** to detect potential fraudulent credit card applications. SOMs are a type of unsupervised learning algorithm used to map high-dimensional data into a low-dimensional grid (often 2D) while preserving the topological structure of the data.

This repository contains a Python script that reads credit card application data, trains a SOM to detect patterns and clusters, visualizes the results, and identifies potential fraud cases.

---

## Hardware Requirements

To run the SOM code efficiently, your machine should meet the following hardware specifications:

- **Processor**: Intel Core i5 (or equivalent) and above.
- **RAM**: Minimum 4GB (8GB or higher recommended for larger datasets).
- **Storage**: At least 1GB free space for the dataset, libraries, and results.
- **GPU (Optional)**: If working with large datasets or more complex models, a GPU like NVIDIA GeForce GTX 1050 or higher will help accelerate computations.

---

## Software Requirements

The following software dependencies are necessary to run the SOM code:

### Operating System
- Windows, macOS, or Linux

### Python Version
- Python 3.7+

### Python Libraries
The code requires the following Python libraries to be installed:

1. NumPy: For numerical and matrix operations.
2. Pandas: To load and manipulate the dataset.
3. Matplotlib: For plotting and visualizing the SOM results.
4. scikit-learn: For feature scaling.
5. MiniSom: A lightweight library to implement Self-Organizing Maps. 
   ```bash
   pip install numpy
   pip install pandas
   pip install matplotlib
   pip install scikit-learn
   pip install minisom

   ```
this code can be run on google colab and only minisom library has to be installed in the colab notebook while on other IDE have to install all the 5 packages for the code to run

## How to Run

-clone the repository

```bash
git clone https://github.com/Adithya2-2/SOM-implementation-DA-2-Big-data-analytics.git
```
-install dependencies 
-install the required python libraries using pip

```bash
pip install numpy pandas matplotlib scikit-learn minisom

```

-Run the Script
-Execute the python script
```bash
python SOM.py
```


