import pandas as pd
import pytest
import helper_fns

# functions are special functions pytest keeps track of to safely share resources and/or resource definitions
@pytest.fixture()
def df1():
    data = {'Name':['Eva', 'Ben', 'Ugo', 'Smoky'], 'marks':[99, 98, 95, 90]} 
    return pd.DataFrame(data, index =['rank1', 'rank2', 'rank3', 'rank4'])

@pytest.fixture()
def df2():
    data = {'Name':['Messi', 'Ronaldo', 'Rukky', 'Michael'], 'marks':[199, -29, 95, 90]} 
    return pd.DataFrame(data, index =['rank1', 'rank2', 'rank3', 'rank4'])

def test_id_checker(df1, df2):
    assert(type(df1) == type(df2))

def test_nan_checker(df1, df2):
    assert(type(df1) == type(df2))

def test_car_var_checker(df1, df2):
    assert(type(df1) == type(df2))