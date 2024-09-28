## Credit Card Applications Dataset

This dataset was obtained from the **[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/143/statlog+australian+credit+approval)**.

### Description
This dataset concerns credit card applications. All attribute names and values have been changed to meaningless symbols to protect the confidentiality of the data.

This dataset is interesting because it contains a mix of attributes—both continuous and nominal with varying numbers of values. Additionally, there are a few missing values. 

There are:
- **6 numerical** attributes
- **8 categorical** attributes

The labels have been modified for statistical algorithms. For instance, attribute 4 originally had the labels `p`, `g`, and `gg`, which have been changed to `1`, `2`, and `3`.

---

### Attribute Information

| **Attribute** | **Type**            | **Original Labels**    | **New Labels**                     |
|---------------|---------------------|------------------------|-------------------------------------|
| **A1**        | Categorical          | (formerly: a, b)        | 0, 1                               |
| **A2**        | Continuous           | -                      | -                                  |
| **A3**        | Continuous           | -                      | -                                  |
| **A4**        | Categorical          | (formerly: p, g, gg)    | 1, 2, 3                            |
| **A5**        | Categorical          | (formerly: ff, d, i, k, j, aa, m, c, w, e, q, r, cc, x) | 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 |
| **A6**        | Categorical          | (formerly: ff, dd, j, bb, v, n, o, h, z) | 1, 2, 3, 4, 5, 6, 7, 8, 9         |
| **A7**        | Continuous           | -                      | -                                  |
| **A8**        | Categorical          | (formerly: t, f)        | 1, 0                               |
| **A9**        | Categorical          | (formerly: t, f)        | 1, 0                               |
| **A10**       | Continuous           | -                      | -                                  |
| **A11**       | Categorical          | (formerly: t, f)        | 1, 0                               |
| **A12**       | Categorical          | (formerly: s, g, p)     | 1, 2, 3                            |
| **A13**       | Continuous           | -                      | -                                  |
| **A14**       | Continuous           | -                      | -                                  |
| **A15**       | Class Attribute      | (formerly: +, -)        | 1, 2                               |

---

### Highlights
- **Mixed Attributes**: This dataset contains a variety of data types, including continuous values and categorical variables with both small and large sets of values.
- **Transformed Labels**: To ease the process of statistical analysis, the original labels were transformed into numerical values.
- **Missing Values**: A few attributes may have missing values.
  
---

