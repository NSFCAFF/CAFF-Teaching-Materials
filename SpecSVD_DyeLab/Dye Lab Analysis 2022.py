# The following code is for the SVD analysis portion of the NSF CAFF lab "How many dyes are in a collection of
# colorful samples?"

# Instructors should include this code in each student folder as a Jupyter Notebook on the course JupyterHub
# Comments can be included as markdown cells to help students understand each step in the analysis (see example of formatting on Git)


# ------ Beginning of Analysis Code ------


# Applying Singular Value Decomposition to your Unknown Mixture Data
# The following code will help you determine the number of significant singular values associated with your data,
# which roughly corresponds to how many unique absorbance spectra make up your total mixture spectrum.
# This will help tell you how many unique dyes you have in your mixture.


# (1) Importing Python Libraries: This first cell will import the Python libraries you need to do your data analysis.
# A Python library contains collections of prewritten code that can be used to perform tasks without having to re-write
# code yourself. This allows you to simplify code you are writing.


from matplotlib import pyplot as plt
from scipy import linalg
import numpy as np
import pandas as pd


# 2) Import your formatted absorbance data: Change the file path found in the second Jupyter cell (the default should
# read "INSERT_FILE_NAME.csv". Then, run this second Jupyter cell to import your data from the .csv file you uploaded.
# You’ll be doing this with the Python library Pandas, which we renamed “pd” when we imported it (for simplicity).
#
# Notice that for this cell, we see an output. This is because we asked our code to print our imported data, using
# “print(abs_data)”. When programming, you do not have to print your data, but sometimes we do this to understand what
# our code is doing (here, to directly see what we’re importing).


# Import .csv of absorbance data
abs_data = pd.read_csv(r'INSERT_FILE_NAME.csv', header=None)
print(abs_data)


# (3) Perform singular value decomposition: Run the third cell to compute the singular values associated with your data.
# You’ll be doing this with the Python library SciPy.


# Compute SVD
U, s, V = linalg.svd(abs_data, full_matrices=False)
print(s)


# (4) Plot your data (part 1): Run the fourth cell to plot your singular values using the Python library matplotlib.


# Plot the singular value spectrum
plt.plot(np.arange(1, s.shape[0] + 1),
         s,
         c='k',
         marker='o',
         linestyle='--')
plt.xlabel('Singular Value')
plt.ylabel('SV Magnitude')
plt.title('Singular Value (SV) Spectrum')
plt.show()
print('Singular values=', s)


# (5) Plot your data (part 2): Run the fifth cell to plot your singular values on a log scale (to better see differences
# in small values) using the Python library matplotlib.


# Plot the log-scaled singular value spectrum (for better comparison of smaller singular values)
plt.plot((np.arange(1, s.shape[0] + 1)),
         np.log(s),
         c='k',
         marker='o',
         linestyle='--')
plt.xlabel('Singular Value')
plt.ylabel('Log-scaled SV Magnitude')
plt.tick_params(axis='x')
plt.tick_params(axis='y')
plt.title('Log-scaled Singular Value (SV) Spectrum')
plt.show()
print('log-scaled singular values=', np.log(s))


# (4b and 5b) Save your plots above by either:
#
#   (i)  right clicking on the plot and selecting “Save Image As…”
#
#   (ii) copying your image into a Word document or your Excel file and saving it there
#
#   (iii) or taking a screen shot of your plots.



# (6) Save your singular values to a .csv file: Change the filename found in the final cell to indicate which set of
# singular values you are calculating (e.g., “Group099_SV.csv” or “UVVis_SV.csv”). Run the final cell to save your
# singular values as a .csv file using the python library numpy.



# Save singular values as a .csv file
SV = np.array(s)
np.savetxt("SingularValues.csv", SV)