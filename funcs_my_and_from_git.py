import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import math

def showTypes(columns, vvv=False):
  types = dict()
  for feature in columns:
    type_ = data_raw[feature].dtype.name
    if type_ not in types.keys():
      types[type_] = []
    types[type_].append(feature)
    if vvv:
      print(f"{type_} it's {feature}")

  for key, val in types.items():
    print(f'{key} -> {len(val)}')

def showSchedule(data_raw, group='Category', title_text=""):
  title_prev_text = 'Категории мотоциклов '
  if group != 'Category':
    title_prev_text = ''
  data = data_raw.groupby(group).count()
  fig, ax = plt.subplots()
  ax = sns.barplot(x=data.index.tolist(), y=data.max(axis=1),palette = "YlGn")
  ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
  ax.set_title("Категории мотоциклов " + title_text)

def bar_plot_series(series, title=None):
  fig, ax = plt.subplots()
  with sns.axes_style("darkgrid"):
      ax = sns.barplot(x=series.index.tolist(), y=series.values, ax=ax, palette = "YlGn")
      ax.set_xticks(ax.get_xticks(), ax.get_xticklabels(), rotation=45, ha='right')
  if title:
      ax.set_title(title)
  return fig, ax

def randomRaws(data, n_indices):
  indices = data.index.tolist()
  if len(indices) <= n_indices:
      return data.copy()
  rnd_indices = np.random.choice(indices, size=n_indices, replace=False)
  rnd_rows = data.loc[rnd_indices,:]
  return rnd_rows

def showDiversity(data, name):
  print(f"Разнообразие признака {name}: {data[name].unique().shape[0]} вариантов")

def n_cylinders(word):
  match word:
    case 'Single cylinder':
      return 1
    case 'V2', 'Twin', 'Two cylinder boxer':
      return 2
    case 'In-line three', 'Diesel':
      return 3
    case 'V4', 'In-line four', 'Four cylinder boxer':
      return 4
    case 'In-line six', 'Six cylinder boxer':
      return 6
    case 'Electric':
      return 0

def find_keyword_return_value(s, mapping):
  if type(s) != str:
    s = str(s)
  s = s.lower()
  for k in mapping.keys():
    if k in s:
        return mapping[k]
  return np.nan

def subfeature_from_feature(data, col_name, new_column, mapping):
  data[new_column] = data.loc[:,col_name].apply(lambda x: find_keyword_return_value(x, mapping))
  return data

def unify_pistons(s):
    s = s.lower()
    new = None
    if re.findall(r'([0-9]+)\s+(pistons?)\b',s):
        new = re.sub(r'([0-9]+)\s+(pistons?)\b', '\\1piston',s)
    elif re.findall(r'([0-9]+)-(pistons?)\b',s):
        new = re.sub(r'([0-9]+)-(pistons?)\b', '\\1piston',s)
    numkeys = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
    nums = {k:v+1 for v,k in enumerate(numkeys)}
    nums['twin'] = 2
    for numstring, num in nums.items():
        pattern1 = r'(' + numstring + r')\s+(pistons?)\b'
        pattern2 = r'(' + numstring + r')-(pistons?)\b'
        pattern = None
        if re.findall(pattern1,s):
            pattern = pattern1
        elif re.findall(pattern2,s):
            pattern = pattern2
        substitution_1, substitution_2 = r'(' + numstring + r')', str(num)
        if pattern:
            new = re.sub(pattern, '\\1piston',s)
            new = re.sub(substitution_1, substitution_2, new)
    return new if new else s

def brakes_build(data, column, key_word):
  counts = data.groupby(column).count()
  unique_categories_count = pd.Series(counts.iloc[:,0], index=counts.index)

  data[column] = data.loc[:,column].apply(lambda x: unify_pistons(x))

  data = subfeature_from_feature(data, column,f'n_disks_{key_word}', {'drum':0,'single':1,'double':2})
  data = subfeature_from_feature(data, f'n_disks_{key_word}', f'has_{key_word}_disk', {'0':0,'1':1,'2':1})
  data = subfeature_from_feature(data, column, f'n_pistons_{key_word}', {'1piston':1,'2piston':2,'3piston':3,'4piston':4, '6piston':6})
  data = subfeature_from_feature(data, column, f'abs_{key_word}', {'abs':1})

  data[f'abs_{key_word}'] = data[f'abs_{key_word}'].fillna(value=0)
  data[f'n_pistons_{key_word}'] = data[f'n_pistons_{key_word}'].fillna(value=1)

# Tires/Tyres

tyres_data = pd.read_csv('Tyre_sizes.csv', sep='\t', index_col=0)

def inch_to_mm(f):
  return f * 25.4

def apply_function_elementwise(values, func):
  for val in values:
    yield func(val)

def parse_numeric_tyre(s):
  width, diameter = re.findall(r'(\d*[\.]?\d*)-(\d+)', s)[0]
  width, diameter = apply_function_elementwise([width, diameter], float)
  height = np.nan
  construction = 'B'
  return width, height, construction, diameter

def clean_tyre_description(s):
  s = s.upper()
  new_s = ''
  for c in s:
    if c.isalnum():
      new_s += c
    else:
      for special in [r'-', r'/', '.']:
        if c == special:
            new_s += c
  return new_s

def fill_tyre_width(data, col_name):
  unique_failed_classes = data[data[col_name].isnull()]['Category'].unique()
  for category in unique_failed_classes:
    avg = data.loc[data['Category']==category, col_name].mean()
    data.loc[data['Category']==category,col_name] = data.loc[data['Category']==category, col_name].fillna(value=avg)
  return data

def fill_tyre_height(data, col_name):
  width_column = col_name.replace('height', 'width')
  unique_failed_classes = data[data[col_name].isnull()]['Category'].unique()
  for category in unique_failed_classes:
    cat_filter = data['Category']==category
    avg_height = data.loc[cat_filter, col_name].mean()
    avg_width = data.loc[cat_filter, width_column].mean()
    avg_ratio = avg_height / avg_width
    filler = data.loc[cat_filter, col_name].fillna(data.loc[cat_filter, width_column] * avg_ratio)
    data.loc[cat_filter, col_name] = filler
  return data

def fill_tyre_diameter(data, col_name):
  unique_failed_classes = data[data[col_name].isnull()]['Category'].unique()
  for category in unique_failed_classes:
    avg = data.loc[data['Category']==category, col_name].mean()
    data.loc[data['Category']==category,col_name] =\
      data.loc[data['Category']==category, col_name].fillna(value=avg)
  return data

def parse_alphanumeric_tyre(s):
  alphas = ['H', 'J', 'M', 'N', 'P', 'R', 'T', 'U', 'V']
  nums = [80,90,100,110,110,120,130,140,150]
  alphanum_width = {k:v for (k,v) in zip(alphas, nums)}
  width, ratio, construction, diameter  = re.findall(r'(M[A-Z]+)(\d+)-([A-Z])(\d+)', s)[0]
  width = (alphanum_width[width[1]])
  height = width * float(ratio) / 100
  diameter = inch_to_mm(float(diameter))
  return width, height, construction, diameter

def tyre_speed_and_construction(keys='upper'):
  labels = tyres_data.index.tolist()
  for i, label in enumerate(labels):
    if label == 'ZR':
      labels[i] = 'Z'
  speeds = tyres_data.iloc[:,0].tolist()
  if keys == 'lower':
    labels = [label.lower() for label in labels]
  return {k:v for (k,v) in zip(labels, speeds)}

def parse_standard_tyre(s):
  width, ratio, speed_construction, diameter = re.findall(r'([0-9]+)/([0-9]+)\-([A-Z]+)?([0-9]+)',s)[0]
  speed_dict = tyre_speed_and_construction()
  if len(speed_construction) > 0:
    if len(speed_construction) > 1:
      speed = speed_construction[0] if speed_construction[0] in speed_dict.keys() and speed_construction[0] not in ['B','R'] else 'A'
      construction = speed_construction[1] if speed_construction[1] in ['B','R'] else 'B'
      if speed == 'A' and construction == 'B':
        speed = speed_construction[1] if speed_construction[1] in speed_dict.keys() and speed_construction[1] not in ['B','R'] else 'A'
        construction = speed_construction[0] if  speed_construction[0] in ['B','R'] else 'B'
    else:
      letter = speed_construction
      if letter in speed_dict.keys() and letter not in ['B','R']: # This means the letter refers to speed
        speed = letter
        construction = 'B' # Construction is B by default
      elif letter in ['R', 'B']:
        construction = letter
        speed = 'A' # Equivalent to not indicated
      else:   # Found another letter that does not belong to any known classification
        construction = 'B'
        speed = 'A'
  else:
    construction, speed = 'B', 'A'

  return width, ratio, speed, construction, diameter


def get_tyre_data(s):
  try:
    width, ratio, speed, construction, diameter = parse_standard_tyre(s)            
    width, ratio, diameter = apply_function_elementwise([width, ratio, diameter], float)
    height = width * ratio /100
    diameter = inch_to_mm(diameter) if diameter < 100 else diameter
    label_format = 'I'
  except:
    try:    #Numeric
      width, height, construction, diameter = parse_numeric_tyre(s)
      # Convert to inches except when the value is so big it must be already in mm
      width = inch_to_mm(width) if width < 10 else width
      diameter = inch_to_mm(diameter) if diameter < 50 else diameter
      speed = 'A'
      label_format = 'N'
    except:
      try:
        width, height, construction, diameter = parse_alphanumeric_tyre(s)
        diameter = inch_to_mm(diameter) if diameter < 50 else diameter
        speed = 'A'
        label_format = 'A'
      except IndexError:
        width, height, speed, construction, diameter, label_format = np.full((6,), np.nan)
  return pd.Series([width, height, speed, construction, diameter, label_format])

def tyres_columns(data, column, key_word):
  data[column] = data.loc[:,column].apply(clean_tyre_description)
  new_columns, tyre_ptys = [], ['width', 'height', 'speed', 'construction', 'diameter', 'label_format']
  for property in tyre_ptys:
    new_columns.append(key_word + '_tire_' + property)
  data[new_columns] = data[column].apply(get_tyre_data)
  return data

def binarize_categorical(feature):
  """Receives a vector of a categorical feature.
  Returns the matrix of the binarized feature and the names of each category
  (which are the columns in the matrix). If the feature has just two classes,
  return one binary column"""
  unique_vals = feature.unique()
  if len(unique_vals) == 2:
    bin_array, unique_vals = binarize_dual(feature)
  else:
    bin_array = np.zeros((len(feature), len(unique_vals)))
    for j, cat in enumerate(unique_vals):
      bin_array[np.where(feature==cat),j] = 1
      bin_array[np.where(feature!=cat),j] = 0
  return bin_array, unique_vals

def binarize_dual(feature):
  """Receives a feature column with binary labels (only two classes)
  Returns a single feature column of zeros and ones and the name of the class encoded with 1 (as a list)"""
  unique_vals = feature.unique()
  cat = unique_vals[0]
  bin_array = np.zeros((len(feature), ))
  bin_array[np.where(feature==cat)] = 1
  bin_array[np.where(feature!=cat)] = 0
  return bin_array, [cat]

def feature_to_binary(data, col_name):
  """Create binary vectors corresponding to each possible category in the feature
  represented by the column passed.
  Return the received DataFrame, with the new columns which are the binary arrays"""
  feature = data.loc[:,col_name]
  bin_array, names = binarize_categorical(feature)
  names = [col_name+'_'+str(n) for n in names]
  if len(names) > 1:
    bin_array_t = bin_array.T
    for i, col_t in enumerate(bin_array_t):
      data.loc[:,names[i]] = col_t.T
  else:
    data.loc[:,names[0]] = bin_array.T
  return data

def tires_build(data, key_word):
  # Width
  data = fill_tyre_width(data, f'{key_word}_tire_width')
  # Height
  data = fill_tyre_height(data, f'{key_word}_tire_height')
  # Speed
  data.loc[:, f'{key_word}_tire_speed'] = data.loc[:, f'{key_word}_tire_speed'].fillna(value='A')

  # Construction
  data.loc[:, f'{key_word}_tire_construction'] = data.loc[:, f'{key_word}_tire_construction'].fillna(value='B')

  # Diameter
  data = fill_tyre_diameter(data, f'{key_word}_tire_diameter')
  # Speed to numeric
  speed_dict = tyre_speed_and_construction(keys='lower')
  avg = sum(speed_dict.values()) / len(speed_dict)
  data = subfeature_from_feature(data, f'{key_word}_tire_speed', f'{key_word}tire_speed', speed_dict)
  data.loc[:, f'{key_word}tire_speed'] = data.loc[:, f'{key_word}tire_speed'].fillna(value=avg)
  data.drop(columns=[f'{key_word}_tire_speed'], axis=1, inplace=True)

  # Construction to binary
  data = feature_to_binary(data, f'{key_word}_tire_construction')
  data.drop(columns=[f'{key_word}_tire_construction'], axis=1, inplace=True)

  # Format to binary
  data[f'{key_word}_tire_label_format'] = data[f'{key_word}_tire_label_format'].fillna(value='I')
  data = feature_to_binary(data, f'{key_word}_tire_label_format')
  data.drop(columns=[f'{key_word}_tire_label_format'], axis=1, inplace=True)