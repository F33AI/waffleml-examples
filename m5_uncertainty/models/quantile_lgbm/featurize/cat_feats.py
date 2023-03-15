import pandas as pd
from .utils import oh_transform


def store_ohe(store_id: pd.Series) -> pd.DataFrame:
    names = ['CA_1', 'CA_2', 'CA_3', 'CA_4', 'TX_1', 'TX_2', 'TX_3', 'WI_1',
             'WI_2', 'WI_3']
    return oh_transform(store_id, names)


def item_ohe(item_id: pd.Series) -> pd.DataFrame:
    names = ['FOODS_3_215', 'FOODS_3_449', 'FOODS_3_555', 'FOODS_3_586',
             'HOBBIES_1_150', 'HOBBIES_1_151', 'HOBBIES_1_169', 'HOBBIES_1_179',
             'HOUSEHOLD_1_118', 'HOUSEHOLD_1_516', 'HOUSEHOLD_1_521']
    return oh_transform(item_id, names)


def month_ohe(month: pd.Series) -> pd.DataFrame:
    names = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    return oh_transform(month, names)


def year_ohe(year: pd.Series) -> pd.DataFrame:
    names = [2011, 2012, 2013, 2014, 2015, 2016]
    return oh_transform(year, names)


def state_ohe(state_id: pd.Series) -> pd.DataFrame:
    names = ['WI', 'CA', 'TX']
    return oh_transform(state_id, names)


def cat_ohe(cat_id: pd.Series) -> pd.DataFrame:
    names = ['FOODS', 'HOBBIES', 'HOUSEHOLD']
    return oh_transform(cat_id, names)


def dept_ohe(dept_id: pd.Series) -> pd.DataFrame:
    names = ['HOBBIES_1', 'HOBBIES_2', 'HOUSEHOLD_1', 'HOUSEHOLD_2', 'FOODS_1',
             'FOODS_2', 'FOODS_3']
    return oh_transform(dept_id, names)


def event_name_1_ohe(event_name_1: pd.Series) -> pd.DataFrame:
    names = ['nan', 'SuperBowl', 'ValentinesDay', 'PresidentsDay', 'LentStart',
             'LentWeek2', 'StPatricksDay', 'Ramadan starts', 'OrthodoxEaster',
             'LaborDay', 'Cinco De Mayo', 'MartinLutherKingDay', 'MemorialDay',
             'NBAFinalsStart', 'ColumbusDay', "Father's day", 'IndependenceDay',
             'Purim End', 'Eid al-Fitr', 'Pesach End', "Mother's day", 'Easter',
             'EidAlAdha', 'VeteransDay', 'Thanksgiving', 'Christmas', 'NewYear',
             'OrthodoxChristmas', 'NBAFinalsEnd', 'Chanukah End', 'Halloween']

    return oh_transform(event_name_1, names)


def event_name_2_ohe(event_name_2: pd.Series) -> pd.DataFrame:
    names = ['nan', 'Easter', 'Cinco De Mayo', 'OrthodoxEaster', "Father's day"]
    return oh_transform(event_name_2, names)


def event_type_1_ohe(event_type_1: pd.Series) -> pd.DataFrame:
    names = ['nan', 'Sporting', 'Cultural', 'National', 'Religious']
    return oh_transform(event_type_1, names)


def event_type_2_ohe(event_type_2: pd.Series) -> pd.DataFrame:
    names = ['nan', 'Cultural', 'Religious']
    return oh_transform(event_type_2, names)


def wday_ohe(wday: pd.Series) -> pd.DataFrame:
    names = [1, 2, 3, 4, 5, 6, 7]
    return oh_transform(wday, names)

