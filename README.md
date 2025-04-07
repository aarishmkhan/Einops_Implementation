# Einops Rearrange – NumPy Implementation

## Assignment Goal

This project implements a custom `rearrange` function in pure NumPy, mimicking the core functionality of the `einops.rearrange` operation. The goal is to provide flexible tensor reordering using a concise pattern string, without relying on the actual Einops library. The implementation supports common tensor transformations such as reshaping (merging or splitting axes), transposition of axes, adding/repeating axes, and handling the ellipsis (`...`) notation for batch dimensions.

## Approach

### Pattern Parsing
- The function interprets a pattern string of the form `input_pattern -> output_pattern`.
- A custom parser processes each side of the `->`:
  - It identifies individual axis labels, grouped axes in parentheses (for splitting or merging), numeric axes (e.g. `2` meaning create a new axis of length 2), and the ellipsis `...` representing unspecified dimensions.
- The parser validates the pattern (checking for duplicates or invalid names) and builds a structured representation of the desired input and output axes.

### Shape Calculation
- Using the parsed pattern and any provided `axes_lengths` arguments (for specifying sizes of symbols or new axes), the code computes how to transform the input array’s shape into the output shape. This includes determining:
  - If any axes need to be combined or split (and the intermediate shapes for those operations).
  - The necessary permutation of axes (for reordering dimensions).
  - Where new axes of size 1 should be inserted and how they should be repeated to the target size.

### Applying Transformations
- The input NumPy array is then transformed in steps:
  - **Reshape** – The array is reshaped to an initial shape that splits or merges axes as required by the pattern.
  - **Transpose** – The axes are rearranged (if needed) according to the output order specified by the pattern.
  - **Add/Repeat Axes** – If the output pattern contains new axes not in the input (for example, a dimension labeled with a size via `axes_lengths`), those axes are inserted (as size 1) and then repeated (tiled) to the specified length.
  - **Final Reshape** – A final reshape is applied to reach the exact output shape (especially after splitting or merging operations).

Throughout this process, the implementation uses NumPy operations (`reshape`, `transpose`, and `np.tile` for repetition) to manipulate the tensor. The parser results are cached internally to avoid re-parsing the same pattern string on repeated calls, and consecutive reshape operations are optimized when possible to minimize overhead.

## Key Design Decisions

- **Custom Parser & Data Structures**:  
  A `ParsedExpression` class is used to represent the parsed pattern. This helps encapsulate logic for handling parentheses groups and ellipses, and makes it easier to reuse and even cache pattern parsing results for performance.

- **Explicit Error Handling**:  
  A custom `EinopsError` exception class provides clear error messages. The code checks for issues like missing `->` in the pattern, duplicate axis names, mismatched tensor dimensions vs. pattern, or missing `axes_lengths` for unknown sizes, and raises informative errors in those cases.

- **Axis Name Rules**:  
  The implementation forbids invalid axis identifiers (e.g. Python keywords or names starting with numbers) and treats an axis of length 1 specified by a numeric label as a no-op (since it doesn’t change the tensor shape). These choices mirror Einops conventions and prevent ambiguous patterns.

- **Performance Considerations**:  
  Pattern parsing results are cached using `functools.lru_cache`. In addition to caching the parsed patterns, the code merges certain operations. For example, if two axes are created and then immediately merged in the sequence, it will combine those steps. This reduces the number of intermediate array creations, keeping the rearrangement as efficient as possible given the pure NumPy context. 

## Example Usage

Below are some usage examples demonstrating the supported operations (similar to the assignment specification):

```python
import numpy as np
# Assume `rearrange` is the implemented function.

# Transpose axes
x = np.random.rand(3, 4)
result = rearrange(x, 'h w -> w h')   # result shape: (4, 3)

# Split an axis into two
x = np.random.rand(12, 10)
result = rearrange(x, '(h w) c -> h w c', h=3)  # splits first axis (12) into 3×4

# Merge axes into one
x = np.random.rand(3, 4, 5)
result = rearrange(x, 'a b c -> (a b) c')  # result shape: (12, 5)

# Repeat an axis (create a new axis of length 4)
x = np.random.rand(3, 1, 5)
result = rearrange(x, 'a 1 c -> a b c', b=4)  # new axis b of length 4 (tile repeat)

# Handle batch dimensions with ellipsis
x = np.random.rand(2, 3, 4, 5)  # shape (2, 3, 4, 5)
result = rearrange(x, '... h w -> ... (h w)')  # combine the last two dimensions
```

Each of these examples uses the pattern string to describe the transformation. For instance, `'(h w) c -> h w c'` takes a tensor of shape `(h*w, c)` and reshapes it to `(h, w, c)` given a specific `h` (here 3) so that `w` is inferred as 4.

## Project Structure

- **Implementation Cells:**  
  The first portion of the notebook contains the implementation of the `rearrange` function along with supporting classes and helper functions. These cells define the custom parser for the einops pattern and the core tensor transformation operations using NumPy.

- **Unit Tests Cells:**  
  Following the implementation, several cells include comprehensive unit tests. These tests cover various use cases such as transposition, axis splitting, merging, repetition, handling ellipsis for batch dimensions, and error handling for invalid patterns or mismatches.

## Running the Notebook and Tests

### Requirements
- Ensure you have Python 3 and NumPy installed.
- No external Einops library is needed.

### Running the Notebook
- Open the `einops.ipynb` notebook in Jupyter or a compatible environment.
- Run all cells (for example, select **Runtime -> Run All**) to execute the implementation and then run the built-in test cases.

### Unit Tests
- The notebook includes a comprehensive suite of unit tests at the end.
- When you run all cells, each test section will output its results and confirm whether the test passed.
- You should see output for various categories (splitting axes, merging axes, combined operations, ellipsis handling, repeating axes, error cases, etc.), with **"✅ Test Passed"** indicators.
- If all tests pass, the implementation is working as expected.
