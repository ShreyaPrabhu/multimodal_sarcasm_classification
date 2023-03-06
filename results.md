### Binary Classification

| Modality | Speaker | Context| Model | F1 Score Positive Class |
| ---------| --------| -------| ------| ------------------------|
| Audio    | No      | No     | SVM   | 67                      |
| Audio    | Yes     | No     | SVM   | 59                      |
| Audio    | No      | No     | GRU   | 66.41                   |
| Audio    | Yes     | No     | GRU   | 69.13                   |
| Audio    | No      | Yes    | SVM   |                         |
| Audio    | Yes     | Yes    | GRU   |                         |

### Multiclass Classification

| Modality | Speaker | Context| Model | F1 Score  |
| ---------| --------| -------| ------| ----------|
| Audio    | No      | No     | SVM   | 46        |
| Audio    | Yes     | No     | SVM   | 46        |
| Audio    | No      | No     | GRU   | 47        |
| Audio    | Yes     | No     | GRU   | 50        |
| Audio    | No      | Yes    | SVM   |           |
| Audio    | Yes     | Yes    | GRU   |           |

