"""
Extracting Rows and Columns using NumPy

This script demonstrates how to extract specific rows, columns,
and individual elements from a NumPy array.

This is essential in Machine Learning for feature selection
and data preprocessing.
"""

import numpy as np

class ArrayExtractor:

    def __init__(self, arr: np.ndarray):
        self.arr = arr

    def extract_last_column(self):
        return self.arr[:, -1:]

    def extract_last_row(self):
        return self.arr[-1:, :]

    def extract_center_element(self):
        rows, cols = self.arr.shape
        return self.arr[rows//2:rows//2+1, cols//2:cols//2+1]


if __name__ == "__main__":

    arr = np.array([
        [1,2,3],
        [4,5,6],
        [7,8,9]
    ])

    extractor = ArrayExtractor(arr)

    print("Last column:\n", extractor.extract_last_column())
    print("\nLast row:\n", extractor.extract_last_row())
    print("\nCenter element:\n", extractor.extract_center_element())
