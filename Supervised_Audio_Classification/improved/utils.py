import csv
import os

base_dir = 'Supervised_Audio_Classification'
data_dir = os.path.join(base_dir, 'data')

train_csv = os.path.join(data_dir, 'train.csv')
eval_csv = os.path.join(data_dir, 'eval.csv')
test_csv = os.path.join(data_dir, 'test_post_competition_scoring_clips.csv')
class_map_csv = os.path.join(data_dir, 'class_map.csv')

# %% class_map
with open(file=train_csv, mode='r') as _rf:
    _reader = csv.DictReader(_rf)
    _unique_labels = sorted(set([_row['label'] for _row in _reader]))

    with open(file=class_map_csv, mode='w', newline='') as _wf:
        _writer = csv.DictWriter(_wf, fieldnames=['class_index', 'class_name'])
        for (_class_index, _class_name) in enumerate(_unique_labels):
            _writer.writerow({'class_index': _class_index, 'class_name': _class_name})

# %% eval.csv
with open(file=test_csv, mode='r') as _rf:
    _rows = [{'fname': _row['fname'], 'label': _row['label'], 'manually_verified': 1} for _row in csv.DictReader(_rf)]

    with open(file=eval_csv, mode='w', newline='') as _wf:
        _writer = csv.DictWriter(_wf, fieldnames=['fname', 'label', 'manually_verified'])
        _writer.writeheader()

        for _row in _rows:
            _writer.writerow(_row)
