import os

my_path = '../scripts/donnees/donnees_5var_month_hour_2tbn_nosplit_clean_subnormalized/'
my_files = os.listdir(my_path)
os.mkdir("arcs_5var_month_hour_2tbn_nosplit_clean_subnormalized")
os.mkdir("parameters_5var_month_hour_2tbn_nosplit_clean_subnormalized")
for f in my_files:
    print('file', f, flush=True)
    # if f[:5] == 'train':
    name = '_'.join(f.split('_')[2:])
    name = name.replace('.csv', '.txt')
    os.system("Rscript apprentissage.R " + my_path + '/' + f +
              " arcs_5var_month_hour_2tbn_nosplit_clean_subnormalized/arcs_" + name +
              " parameters_5var_month_hour_2tbn_nosplit_clean_subnormalized/parameters_" + name)
