import nbformat as nbf

NOTEBOOK_NAME = 'CapsNet.ipynb'
python_file_paths = [
    './model/capsnet.py',
    './model/loss.py',
    './train.py',
]

cells = []
for file_path in python_file_paths:
    with open(file_path, 'r') as f:
        content = f.read()
        cells.append(
            nbf.v4.new_markdown_cell(f'## {file_path}')
        )
        cells.append(
            nbf.v4.new_code_cell(content)
        )


nb = nbf.v4.new_notebook()
nb['cells'] = cells

nbf.write(nb, NOTEBOOK_NAME)
print('Wrote', NOTEBOOK_NAME, 'succesfully')