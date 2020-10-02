import os
import yaml
from genotypes import *

class MyDumper(yaml.Dumper):

    def increase_indent(self, flow=False, indentless=False):
        return super(MyDumper, self).increase_indent(flow, False)

PATH = '/home/zelaa/NIPS19/ANALYSIS/evals_otherseed'


def write_yaml_results(id, results_file, result_to_log):
  results_file = os.path.join('.', results_file+'.yaml')

  try:
    with open(results_file, 'r') as f:
      result = yaml.load(f)
    if id in result.keys():
      result[id].append(result_to_log)
    else:
      result.update({id: [result_to_log]})

    with open(results_file, 'w') as f:
      yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)
  except (AttributeError, FileNotFoundError) as e:
    result = {
        id: [result_to_log]
    }
    with open(results_file, 'w') as f:
      yaml.dump(result, f, Dumper=MyDumper, default_flow_style=False)


def load_results(id, seed):
    filename = os.path.join(PATH, 'eval-EXP-'+str(id), 'ckpt_%d.txt'%seed)
    with open(filename) as f:
        error = [100 - float(x) for x in f.readlines()][-1]
    return error

def main():
    for i, j in zip([10473, 10475, 10477, 10478], [9725, 9726, 9727, 9728]):
        for k in range(1, 4):
            e = load_results(i, k)
            write_yaml_results(i, 'evals', e)
            write_yaml_results(i, 'configs', str(eval('DARTS_%d_%d'%(j, k))))

if __name__ == '__main__':
    main()

