import pandas as pd
from src.slicing.builder import _apply_query

def test_apply_query_simple():
    df = pd.DataFrame({"a":[1,2,3], "b":[0,1,0]})
    mask = _apply_query(df, "a >= 2")
    assert mask.tolist() == [False, True, True]
