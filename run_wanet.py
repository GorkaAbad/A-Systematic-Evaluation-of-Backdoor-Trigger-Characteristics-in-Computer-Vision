import numpy as np
import os

models = ['googlenet', 'resnet', 'vgg', 'alexnet'] # for vgg and tinyimagenet, use sgd instead of adam
#epsilon = np.arange(0.00, 0.021, 0.005) # 5
epsilon = np.arange(0.005, 0.004, 0.005) 
dataname = ['cifar10'] # 3
#dataname = ['tinyimagenet']
setting_dic = {'mnist_vgg':['01:00:00', '5GB', 'short'], 'mnist_googlenet':['00:30:00', '5GB', 'short'], 'mnist_resnet':['01:00:00', '6GB', 'short'], \
              'cifar10_vgg':['02:30:00', '3GB', 'short'], 'cifar10_googlenet':['01:30:00', '3GB', 'short'], 'cifar10_resnet':['02:30:00', '3GB', 'short'], \
              'tinyimagenet_googlenet':['04:00:00', '20GB', 'short'], 'tinyimagenet_vgg':['10:00:00', '20GB', 'medium'], 'tinyimagenet_resnet':['10:00:00', '20GB', 'medium'], 'tinyimagenet_alexnet':['00:30:00', '16GB', 'short'], \
              'cifar10_alexnet':['03:00:00', '4000', 'short'], 'mnist_alexnet':['03:00:00', '4000', 'short']}

seeds = range(0, 1)
template_path = "./template.sh"
for n in seeds:
    for data in dataname[:]:
        for model in models[:]:
            for ep in epsilon[:]:

                template = open(template_path, 'r').read()
                dic_name = '{}_{}'.format(data, model)
                template = template.replace('TIME', setting_dic[dic_name][0])
                template = template.replace('MEM', setting_dic[dic_name][1])
                template = template.replace('TYPE', setting_dic[dic_name][2])
                template = template.replace('EP', str(ep))
                template = template.replace('DATA', str(data))
                template = template.replace('MODELNAME', str(model))
                if data == "tinyimagenet" and model == "vgg":
                    template = template.replace('OPT', 'sgd')
                else:
                    template = template.replace('OPT', 'adam')                        
                template = template.replace('SEED', str(n))
                sbatch_name = "{}/{}_{}_ep_{}.sh".format(n, data, model, ep)
                path = os.path.split(sbatch_name)[0]
                isExist = os.path.exists(path)
                if not isExist:
                    os.makedirs(path)
                with open(sbatch_name, "w") as f:
                    f.write(template)
                #os.system("sbatch {}".format(sbatch_name))

