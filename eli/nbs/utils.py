import pandas as pd
import datetime
import numpy as np
import pickle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def convert_to_atomic(col):
    full_to_atomic = {'Hydrogen' : 'H', 'Helium' : 'He', 'Lithium' : 'Li', 'Beryllium' : 'Be', 'Boron' : 'B', 'Carbon' : 'C', 'Nitrogen' : 'N', 'Oxygen' : 'O', 'Fluorine' : 'F', 'Neon' : 'Ne', 'Sodium' : 'Na', 'Magnesium' : 'Mg', 'Aluminium' : 'Al', 'Silicon' : 'Si', 'Phosphorus' : 'P', 'Sulfur' : 'S', 'Chlorine' : 'Cl', 'Argon' : 'Ar', 'Potassium' : 'K', 'Calcium' : 'Ca', 'Scandium' : 'Sc', 'Titanium' : 'Ti', 'Vanadium' : 'V', 'Chromium' : 'Cr', 'Manganese' : 'Mn', 'Iron' : 'Fe', 'Cobalt' : 'Co', 'Nickel' : 'Ni', 'Copper' : 'Cu', 'Zinc' : 'Zn', 'Gallium' : 'Ga', 'Germanium' : 'Ge', 'Arsenic' : 'As', 'Selenium' : 'Se', 'Bromine' : 'Br', 'Krypton' : 'Kr', 'Rubidium' : 'Rb', 'Strontium' : 'Sr', 'Yttrium' : 'Y', 'Zirconium' : 'Zr', 'Niobium' : 'Nb', 'Molybdenum' : 'Mo', 'Technetium' : 'Tc', 'Ruthenium' : 'Ru', 'Rhodium' : 'Rh', 'Palladium' : 'Pd', 'Silver' : 'Ag', 'Cadmium' : 'Cd', 'Indium' : 'In', 'Tin' : 'Sn', 'Antimony' : 'Sb', 'Tellurium' : 'Te', 'Iodine' : 'I', 'Xenon' : 'Xe', 'Cesium' : 'Cs', 'Barium' : 'Ba', 'Lanthanum' : 'La', 'Cerium' : 'Ce', 'Praseodymium' : 'Pr', 'Neodymium' : 'Nd', 'Promethium' : 'Pm', 'Samarium' : 'Sm', 'Europium' : 'Eu', 'Gadolinium' : 'Gd', 'Terbium' : 'Tb', 'Dysprosium' : 'Dy', 'Holmium' : 'Ho', 'Erbium' : 'Er', 'Thulium' : 'Tm', 'Ytterbium' : 'Yb', 'Lutetium' : 'Lu', 'Hafnium' : 'Hf', 'Tantalum' : 'Ta', 'Tungsten' : 'W', 'Rhenium' : 'Re', 'Osmium' : 'Os', 'Iridium' : 'Ir', 'Platinum' : 'Pt', 'Gold' : 'Au', 'Mercury' : 'Hg', 'Thallium' : 'Tl', 'Lead' : 'Pb', 'Bismuth' : 'Bi', 'Polonium' : 'Po', 'Astatine' : 'At', 'Radon' : 'Rn', 'Francium' : 'Fr', 'Radium' : 'Ra', 'Actinium' : 'Ac', 'Thorium' : 'Th', 'Protactinium' : 'Pa', 'Uranium' : 'U', 'Neptunium' : 'Np', 'Plutonium' : 'Pu', 'Americium' : 'Am', 'Curium' : 'Cm', 'Berkelium' : 'Bk', 'Californium' : 'Cf', 'Einsteinium' : 'Es', 'Fermium' : 'Fm', 'Mendelevium' : 'Md', 'Nobelium' : 'No', 'Lawrencium' : 'Lr', 'Rutherfordium' : 'Rf', 'Dubnium' : 'Db', 'Seaborgium' : 'Sg', 'Bohrium' : 'Bh', 'Hassium' : 'Hs', 'Meitnerium' : 'Mt', 'Darmstadtium' : 'Ds', 'Roentgenium' : 'Rg', 'Copernicium' : 'Cn', 'Nihonium' : 'Nh', 'Flerovium' : 'Fl', 'Moscovium' : 'Mc', 'Livermorium' : 'Lv', 'Tennessine' : 'Ts', 'Og' : 'Oganesson',}
    ### NEED TO CATER TO EDGE CASES
    ### e.g Mag Oxide, Alumina etc.
    ###
    full_name = col
    
    if full_name in full_to_atomic.keys():
        atomic = full_to_atomic[full_name]
        return atomic

    else:
        return np.nan

# WAMEX CLEANING
def replace_neg9999(val):
    if val == -9999: 
        return 0
    else:
        return val
    
    
def abs_halve_neg_val(val):
    if val < 0: 
        return abs(val) / 2
    else: 
        return val

def clean_assays(df):
    df = df.applymap(replace_neg9999)
    df = df.applymap(abs_halve_neg_val)

    # all 9999 -> 0 values are removed
    indexs_0 = df[df.values == 0].index
    df = df.drop(index=indexs_0)
    
    return df

def abundant_ratio(row, abundant_mineral):
    return row / row[abundant_mineral]

def make_pickle_fn(df, model, prefix='../../pickles/'):
    fn = f'{prefix}{type(model).__name__}-'
    for i in df.columns:
        fn = fn + i + '-'

    fn = fn[:-1] + '.pickle'   
    return fn

def filter_by_wamex_minerals(unique_commods, wamex_unique_minerals):
    # filter unique commods by availability in wamex assays
    filtered_unique_commods = []
    for commods in unique_commods:
        filtered_commods = []
        for commod in commods:
            if commod in wamex_unique_minerals:
                filtered_commods.append(commod)
        filtered_unique_commods.append(filtered_commods)
    return filtered_unique_commods

def commod_string(row):
    row['commod_str'] = "-".join(row.dropna().index.to_list()).replace('-Ni', "")
    return row

def load_commod_model(commod_str, model='LinearRegression'):
    with open(f'../../pickles/{model}-{commod_str}-lat-lon.pickle', 'rb') as file:
        m = pickle.load(file)    
    return m

def create_site_pred(commod_group_df):
    '''Gets a dataframe containing a unique commodity group,
    '''
    # drop empty mineral cols
    commod_group_df = commod_group_df.dropna(axis=1)
    # dont know why duplicates
    commod_group_df = commod_group_df.drop_duplicates()
    # create X vars from data.
    commod_group_df_X = commod_group_df.drop(columns=['commod_str', 'SiteCode', 'Ni'])
    # if site already has a Cobalt value, drop.
    if 'Co' in commod_group_df_X.columns:
        commod_group_df_X = commod_group_df_X.drop(columns=['Co'])
    
    commod_str = commod_group_df['commod_str'].iloc[0]
    
    try:
        # use first commod string to find relevant model pickle.
        # Linear Regression
        lr = load_commod_model(commod_str=commod_str, model='LinearRegression')
        lr_preds = lr.predict(commod_group_df_X)
        commod_group_df['Co_pred_lr'] = lr_preds
        # Random Forest
        rf = load_commod_model(commod_str=commod_str, model='RandomForestRegressor')
        rf_preds = rf.predict(commod_group_df_X)
        commod_group_df['Co_pred_rf'] = rf_preds
        
        # make commod_str column
        commod_group_df['commods_used'] = commod_str
            
        site_pred_df = commod_group_df[['SiteCode', 'Co_pred_lr', 'Co_pred_rf', 'commods_used']]

        return site_pred_df
    
    except:
        print(f'Could not find models for {commod_str}')