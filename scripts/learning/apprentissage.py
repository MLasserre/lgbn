import os

my_path = "data/multisample"
my_files = os.listdir(my_path)

for f in my_files:
    print('file', f, flush=True)
    name = '_'.join(f.split('_')[2:])
    name = name.replace('.csv', '.txt')
    print(name)
    # os.system("Rscript apprentissage.R " + my_path + '/' + f +
              # " ../arcs/arcs_" + name +
              # " ../parameters/parameters_" + name)
