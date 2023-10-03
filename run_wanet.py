import os
from typing import Dict, List

# Define constants
MODELS = ['googlenet', 'resnet', 'vgg', 'alexnet'] 
EPSILON = [0.005, 0.010, 0.015, 0.020]
DATANAME = ['tinyimagenet'] 

SETTINGS: Dict[str, List[str]] = {
    'tinyimagenet_googlenet':['0-48:00:00', '128GB', 'short'], 
    'tinyimagenet_vgg':['0-48:00:00', '128GB', 'medium'], 
    'tinyimagenet_resnet':['0-48:00:00', '128GB', 'medium'], 
    'tinyimagenet_alexnet':['0-48:00:00', '128GB', 'short']
}

SEEDS = [42]
TEMPLATE_PATH = "./template.sh"

def replace_template(template: str, data: str, model: str, ep: float, n: int) -> str:
    dic_name = f'{data}_{model}'
    template = template.replace('TIME', SETTINGS[dic_name][0])
    template = template.replace('MEM', SETTINGS[dic_name][1])
    # template = template.replace('TYPE', SETTINGS[dic_name][2])
    template = template.replace('EP', str(ep))
    template = template.replace('DATA', data)
    template = template.replace('MODELNAME', model)
    template = template.replace('OPT', 'sgd' if data == "tinyimagenet" and model == "vgg" else 'adam')                        
    template = template.replace('SEED', str(n))
    return template

def create_directory(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def write_to_file(sbatch_name: str, template: str) -> None:
    with open(sbatch_name, "w") as f:
        f.write(template)

def run_sbatch(sbatch_name: str) -> None:
    os.system("sbatch {}".format(sbatch_name))

def main() -> None:
    for n in SEEDS:
        for data in DATANAME:
            for model in MODELS:
                for ep in EPSILON:
                    with open(TEMPLATE_PATH, 'r') as file:
                        template = file.read()
                    template = replace_template(template, data, model, ep, n)
                    sbatch_name = f"{n}/{data}_{model}_ep_{ep}.sh"
                    path = os.path.split(sbatch_name)[0]
                    create_directory(path)
                    write_to_file(sbatch_name, template)
                    run_sbatch(sbatch_name)

if __name__ == "__main__":
    main()
