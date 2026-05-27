# VNAT Monthly Segment Data Validation

This report validates the raw VNAT monthly segment data before forecasting.

## Check Summary

| check                       | status   | value               |
|:----------------------------|:---------|:--------------------|
| required_columns            | pass     | 7                   |
| date_range_start            | pass     | 2012-01-01 00:00:00 |
| date_range_end              | pass     | 2025-12-01 00:00:00 |
| expected_months             | pass     | 168                 |
| observed_rows               | warn     | 155                 |
| duplicated_months           | pass     | 0                   |
| missing_months              | fail     | 13                  |
| non_positive_values         | pass     | 0                   |
| segment_sum_tolerance       | warn     | 29                  |
| suspicious_repeated_vectors | warn     | 24                  |

## Issue Counts

| issue_type                 |   count |
|:---------------------------|--------:|
| segment_sum_inconsistency  |      29 |
| suspicious_repeated_values |      24 |
| missing_month              |      13 |
