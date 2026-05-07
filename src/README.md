# Black Box Optimisation (BBO) — ML AI Capstone Project

## Notebooks
1. **Check input output files.ipynb**  - checks the number of rows in the input and output data files
2. **BBO input output append.ipynb** - Reads new observations from inputs.txt and outputs.txt and appends them to the existing F1–F8 initial_inputs.npy / initial_outputs.npy files.
3. **BBO12_Enhanced.ipynb** - primary BBO notebook that reads the data in F1–F8 initial_inputs.npy and initial_outputs.npy data files and generates new query points along with metrics and plots


## Data — Inputs and Outputs

All F1-F8 input and output data files (e.g. f1initial_inputs.npy and f1initial_outputs.npy for F1) are named as such and stored in /data folder, with the input dimensions and formats below.

| Function | Input dimensions | Input format |
|----------|-----------------|--------------|
| F1, F2   | 2D              | x1-x2 |
| F3       | 3D              | x1-x2-x3 |
| F4, F5   | 4D              | x1-x2-x3-x4 |
| F6       | 5D              | x1-x2-x3-x4-x5 |
| F7       | 6D              | x1-x2-x3-x4-x5-x6 |
| F8       | 8D              | x1-x2-x3-x4-x5-x6-x7-x8 |

Each input value begins with 0 and is specified to six decimal places (e.g. `0.123456-0.654321` for a 2D function). All outputs are 1D scalar values.

At each iteration, the new query point and its corresponding output are appended to the respective function's data file and used in the subsequent round.

## Run Instructions
1. Upload the F1-F8 input and output data files (e.g. f1initial_inputs.npy and f1initial_outputs.npy for F1) from the '/data' folder in the Jupiter notebook enviornment
2. Upload the **BBO12_Enhanced.ipynb** from 'src' folder and run it to generate the next query point
3. The Input query points are submitted to the Capstone submission page
4. The submission results are provided by the Capstone project leads as 'inputs.txt' and 'output.txt'
5. Use the **BBO input output append.ipynb** to append the data in 'inputs.txt' and 'output.txt' files to F1-F8 input and output data files (e.g. f1initial_inputs.npy and f1initial_outputs.npy for F1)
6. You can then re-run the **BBO12_Enhanced.ipynb** (uploaded from the 'src' folder) to generate the next query point, and continue iteratively using the steps above.