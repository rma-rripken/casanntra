from casanntra.single_or_list import single_or_list
from casanntra.multi_stage_model_builder import *
import pandas as pd
from unittest.mock import MagicMock


import pandas as pd
from unittest.mock import MagicMock


def test_pool_and_align_cases():
    """Unit tests for pool_and_align_cases using a mock MultiStageModelBuilder instance."""
    
    # ✅ Create mock instance
    mock_builder = MagicMock()
    mock_builder.input_names = ['input1', 'input2']
    mock_builder.output_names = ['output1', 'output2']
    
    # ✅ Create sample data with consistent columns + one extra date
    df1 = pd.DataFrame({
        'case': [1, 1, 2, 2, 2],  # Case 2 has an extra date not in df2
        'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02', '2023-01-03']),
        'input1': [10, 20, 30, 40, 50],
        'input2': [1, 2, 3, 4, 5],
        'output1': [100, 200, 300, 400, 500],
        'output2': [None, None, None, None, 2.00]
    })
    
    df2 = pd.DataFrame({
        'case': [1, 1, 3, 3],  # Case 2 is missing in df2
        'datetime': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-01', '2023-01-02']),
        'input1': [15, 25, 35, 45],
        'input2': [5, 6, 7, 8],
        'output1': [None, None, None, None],
        'output2': [150, 250, 350, 450]
    })
    
    # ✅ Run function
    aligned_dfs = MultiStageModelBuilder.pool_and_align_cases(mock_builder, [df1, df2])

    print("df1")
    print(df1)
    print("df2")
    print(df2)

    print("aligned_dfs[0]")
    print(aligned_dfs[0])
    print("aligned_dfs[1]")    
    print(aligned_dfs[1])
    # ✅ Assertions
    assert len(aligned_dfs) == 2, "Should return a list of DataFrames"
    
    pooled_cases = set(aligned_dfs[0]['case'].unique())
    expected_cases = {1, 2, 3}
    assert pooled_cases == expected_cases, f"Cases should be pooled. Expected {expected_cases}, got {pooled_cases}"
    
    pooled_dates = set(aligned_dfs[0]['datetime'].unique())
    expected_dates = set(pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03']))
    assert pooled_dates == expected_dates, f"Dates should be contiguous. Expected {expected_dates}, got {pooled_dates}"
    
    # ✅ Ensure input columns have NO NaN values
    for df in aligned_dfs:
        for col in mock_builder.input_names:
            assert df[col].isna().sum() == 0, f"Input column {col} should never have NaN values"

    # ✅ Ensure output columns have NaNs when missing
    assert aligned_dfs[0]['output2'].isna().sum() > 0, "Missing values should exist for missing outputs"
    assert aligned_dfs[1]['output1'].isna().sum() > 0, "Missing values should exist for missing outputs"

    print("✅ All tests passed!")

# Run the test
test_pool_and_align_cases()
