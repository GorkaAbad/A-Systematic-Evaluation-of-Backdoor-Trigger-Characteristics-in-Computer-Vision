from defenses.Defense import Defense
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
import os
import csv
import torchvision


class NeuralCleanse(Defense):
    """
    Neural Cleanse


    Based on:
    https://github.com/SCLBD/BackdoorBench/blob/main/defense/nc/nc.py
    https://github.com/VinAIResearch/input-aware-backdoor-attack-release/blob/master/defenses/neural_cleanse/neural_cleanse.py

    """

    # Attributes
    # ----------

    lr = None
    init_cost = None
    atk_succ_threshold = None
    early_stop = None
    early_stop_threshold = None
    early_stop_patience = None
    patience = None
    cost_multiplier = None
    epochs = None
    epsilon = None
    n_times_test = None

    # Methods
    # -------

    def __init__(self, args, trainer, attack_id):
        """
        Constructor
        """
        super().__init__(trainer)
        self.lr = args.nc_lr
        self.init_cost = args.nc_init_cost
        self.atk_succ_threshold = args.nc_atk_succ_threshold
        self.early_stop = args.nc_early_stop
        self.early_stop_threshold = args.nc_early_stop_threshold
        self.early_stop_patience = args.nc_early_stop_patience
        self.patience = args.nc_patience
        self.cost_multiplier = args.nc_cost_multiplier
        self.epochs = args.nc_epochs
        self.epsilon = args.nc_epsilon
        self.n_times_test = args.nc_n_times_test
        self.attack_id = attack_id
        trainer_path = args.save_path + '/' + args.model + '_' + args.dataname.upper() + '_' + self.attack_id + '/' + 'trainer.pt'
        self.trainer = torch.load(trainer_path)

    def execute_defense(self):
        """
        Reverse engineer the triggers
        """

        # Be aware that here the input is not transformed, so themask and pattern are smaller
        # TODO: Using the testset
        # if the datset is mnnist
        if self.trainer.dataset.name.lower() == "mnist":
            h, w = self.trainer.dataset.testset.data.shape[1:]
            c = 1
        else:
            h, w, c = self.trainer.dataset.testset.data.shape[1:]
        h = self.trainer.dataset.testset.transform.transforms[1].size
        w = self.trainer.dataset.testset.transform.transforms[1].size

        init_mask = np.ones(
            (1, h, w)).astype(np.float32)
        init_pattern = np.ones(
            (c, h, w)).astype(np.float32)

        for test in range(self.n_times_test):
            print("Test {}:".format(test))

            masks = []
            idx_mapping = {}

            for target_label in range(self.trainer.dataset.n_classes):
                print(
                    "----------------- Analyzing label: {} -----------------".format(target_label))
                recorder = self.train(
                    init_mask, init_pattern, target_label)

                mask = recorder.mask_best
                masks.append(mask)
                idx_mapping[target_label] = len(masks) - 1

            l1_norm_list = torch.stack(
                [torch.sum(torch.abs(m)) for m in masks])
            print("{} labels found".format(len(l1_norm_list)))
            print("Norm values: {}".format(l1_norm_list))
            flag_list, list_mad, bk_model = self.outlier_detection(
                l1_norm_list, idx_mapping)

        self.save_results(None, flag_list, list_mad, bk_model)

    def outlier_detection(self, l1_norm_list, idx_mapping):
        consistency_constant = 1.4826  # From NC paper
        median = torch.median(l1_norm_list)
        mad = consistency_constant * \
            torch.median(torch.abs(l1_norm_list - median))
        min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad

        print("Median: {}, MAD: {}".format(median, mad))
        print("Anomaly index: {}".format(min_mad))

        bk_model = False
        if min_mad < 2:
            print("Not a backdoor model")
        else:
            print("This is a backdoor model")
            bk_model = True

        flag_list = []
        mad_list = []
        for y_label in idx_mapping:
            if l1_norm_list[idx_mapping[y_label]] > median:
                continue
            if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
                mad_list.append(
                    (y_label, torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad))
                flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

        if len(flag_list) > 0:
            flag_list = sorted(flag_list, key=lambda x: x[1])

        print(
            "Flagged label list: {}".format(
                ",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in flag_list]))
        )

        print('List of MAD values: {}'.format(
            ",".join(["{}: {}".format(y_label, l_norm) for y_label, l_norm in mad_list]
                     )))

        return flag_list, mad_list, bk_model

    def train(self, init_mask, init_pattern, target_label):

        test_dataloader = self.trainer.testloader

        # Build regression model
        regression_model = RegressionModel(
            init_mask, init_pattern, self.trainer, self.epsilon).to(self.trainer.device)

        # Set optimizer
        optimizerR = torch.optim.Adam(
            regression_model.parameters(), lr=self.lr, betas=(0.5, 0.9))

        # Set recorder (for recording best result)
        recorder = Recorder(self)

        for epoch in range(self.epochs):
            early_stop = self.train_step(
                regression_model, optimizerR, test_dataloader, recorder, epoch, target_label)
            if early_stop:
                break

        # Save result to dir
        # recorder.save_result_to_dir(opt)

        return recorder

    def train_step(self, regression_model, optimizerR, dataloader, recorder, epoch, target_label):
        print("Epoch {} - Label: {} |:".format(epoch,
                                               target_label))
        # Set losses
        cross_entropy = nn.CrossEntropyLoss()
        total_pred = 0
        true_pred = 0

        # Record loss for all mini-batches
        loss_ce_list = []
        loss_reg_list = []
        loss_list = []
        loss_acc_list = []

        # Set inner early stop flag
        inner_early_stop_flag = False
        for batch_idx, (inputs, labels) in enumerate(dataloader):
            # Forwarding and update model
            optimizerR.zero_grad()

            inputs = inputs.to(self.trainer.device)
            sample_num = inputs.shape[0]
            total_pred += sample_num
            target_labels = torch.ones((sample_num), dtype=torch.int64).to(
                self.trainer.device) * target_label
            predictions = regression_model(inputs)

            loss_ce = cross_entropy(predictions, target_labels)
            loss_reg = torch.norm(regression_model.get_raw_mask(), 2)
            total_loss = loss_ce + recorder.cost * loss_reg
            total_loss.backward()
            optimizerR.step()

            # Record minibatch information to list
            minibatch_accuracy = torch.sum(torch.argmax(
                predictions, dim=1) == target_labels).detach() * 100.0 / sample_num
            loss_ce_list.append(loss_ce.detach())
            loss_reg_list.append(loss_reg.detach())
            loss_list.append(total_loss.detach())
            loss_acc_list.append(minibatch_accuracy)

            true_pred += torch.sum(torch.argmax(predictions,
                                                dim=1) == target_labels).detach()

        loss_ce_list = torch.stack(loss_ce_list)
        loss_reg_list = torch.stack(loss_reg_list)
        loss_list = torch.stack(loss_list)
        loss_acc_list = torch.stack(loss_acc_list)

        avg_loss_ce = torch.mean(loss_ce_list)
        avg_loss_reg = torch.mean(loss_reg_list)
        avg_loss = torch.mean(loss_list)
        avg_loss_acc = torch.mean(loss_acc_list)

        # Check to save best mask or not
        if avg_loss_acc >= self.atk_succ_threshold and avg_loss_reg < recorder.reg_best:
            recorder.mask_best = regression_model.get_raw_mask().detach()
            recorder.pattern_best = regression_model.get_raw_pattern().detach()
            recorder.reg_best = avg_loss_reg
            # recorder.save_result_to_dir(opt)
            print(" Updated !!!")

        # Show information
        print(
            "  Result: Accuracy: {:.3f} | Cross Entropy Loss: {:.6f} | Reg Loss: {:.6f} | Reg best: {:.6f}".format(
                true_pred * 100.0 / total_pred, avg_loss_ce, avg_loss_reg, recorder.reg_best
            )
        )

        # Check early stop
        if self.early_stop:
            if recorder.reg_best < float("inf"):
                # print(recorder.reg_best, recorder.early_stop_reg_best, self.early_stop_threshold,
                #       self.early_stop_threshold * recorder.early_stop_reg_best)
                if recorder.reg_best >= self.early_stop_threshold * recorder.early_stop_reg_best:
                    recorder.early_stop_counter += 1
                    print("Early_stop_counter: {}".format(
                        recorder.early_stop_counter))

                else:
                    recorder.early_stop_counter = 0

            recorder.early_stop_reg_best = min(
                recorder.early_stop_reg_best, recorder.reg_best)

            if (
                recorder.cost_down_flag
                and recorder.cost_up_flag
                and recorder.early_stop_counter >= self.early_stop_patience
            ):
                print("Early_stop !!!")
                inner_early_stop_flag = True

        if not inner_early_stop_flag:
            # Check cost modification
            if recorder.cost == 0 and avg_loss_acc >= self.atk_succ_threshold:
                recorder.cost_set_counter += 1
                if recorder.cost_set_counter >= self.patience:
                    recorder.reset_state(self)
            else:
                recorder.cost_set_counter = 0

            if avg_loss_acc >= self.atk_succ_threshold:
                recorder.cost_up_counter += 1
                recorder.cost_down_counter = 0
            else:
                recorder.cost_up_counter = 0
                recorder.cost_down_counter += 1

            if recorder.cost_up_counter >= self.patience:
                recorder.cost_up_counter = 0
                print("Up cost from {} to {}".format(recorder.cost,
                                                     recorder.cost * recorder.cost_multiplier_up))
                recorder.cost *= recorder.cost_multiplier_up
                recorder.cost_up_flag = True

            elif recorder.cost_down_counter >= self.patience:
                recorder.cost_down_counter = 0
                print("Down cost from {} to {}".format(recorder.cost,
                                                       recorder.cost / recorder.cost_multiplier_down))
                recorder.cost /= recorder.cost_multiplier_down
                recorder.cost_down_flag = True

            # Save the final version
            if recorder.mask_best is None:
                recorder.mask_best = regression_model.get_raw_mask().detach()
                recorder.pattern_best = regression_model.get_raw_pattern().detach()

        return inner_early_stop_flag

    def save_results(self, path=None, flag_list=[], list_mad=[], bk_model=False) -> None:

        super().save_results(path)

        if path is None:
            path = self.trainer.save_path

        path_csv = self.get_path(path)

        # Write the results to the csv file
        header = ['id', 'attack_id', 'dataset', 'model', 'lr', 'is_bk',
                  'init_cost', 'atk_threshold', 'epochs', 'seed',
                  'flag_list', 'list_mad']

        if not os.path.exists(path_csv):
            with open(path_csv, 'w') as f:
                writer = csv.writer(f)
                writer.writerow(header)

        # Use the defense's id for the folder's name as the trainer id will be
        # the same with the attack id which will lead to overwritting the
        # folder of the previous experiment.
        self.trainer.id = self.id

        with open(path_csv, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.id,
                self.attack_id,
                self.trainer.dataset.name,
                self.trainer.model.name,
                self.lr,
                bk_model,
                self.init_cost,
                self.atk_succ_threshold,
                self.epochs,
                self.trainer.seed,
                flag_list,
                list_mad
            ])


class Recorder:
    def __init__(self, nc):
        super().__init__()

        # Best optimization results
        self.mask_best = None
        self.pattern_best = None
        self.reg_best = float("inf")

        # Logs and counters for adjusting balance cost
        self.logs = []
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # Counter for early stop
        self.early_stop_counter = 0
        self.early_stop_reg_best = self.reg_best

        # Cost
        self.cost = nc.init_cost
        self.cost_multiplier_up = nc.cost_multiplier
        self.cost_multiplier_down = nc.cost_multiplier ** 1.5

    def reset_state(self, nc):
        self.cost = nc.init_cost
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False
        print("Initialize cost to {:f}".format(self.cost))

    def save_result_to_dir(self, opt):
        result_dir = os.path.join(opt.result, opt.dataset)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, opt.attack_mode)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)
        result_dir = os.path.join(result_dir, str(opt.target_label))
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        pattern_best = self.pattern_best
        mask_best = self.mask_best
        trigger = pattern_best * mask_best

        path_mask = os.path.join(result_dir, "mask.png")
        path_pattern = os.path.join(result_dir, "pattern.png")
        path_trigger = os.path.join(result_dir, "trigger.png")

        torchvision.utils.save_image(mask_best, path_mask, normalize=True)
        torchvision.utils.save_image(
            pattern_best, path_pattern, normalize=True)
        torchvision.utils.save_image(trigger, path_trigger, normalize=True)


class RegressionModel(nn.Module):

    epsilon = None

    def __init__(self,  init_mask, init_pattern, trainer, epsilon):

        super(RegressionModel, self).__init__()
        self.epsilon = epsilon
        self.trainer = trainer
        self.mask_tanh = nn.Parameter(
            torch.tensor(init_mask), requires_grad=True)
        self.pattern_tanh = nn.Parameter(
            torch.tensor(init_pattern), requires_grad=True)

        self.classifier = trainer.model.model.to(trainer.device)
        self.classifier.eval()
        self.normalizer = self.get_normalizer()
        self.denormalizer = self.get_denormalizer()

    def forward(self, x):
        mask = self.get_raw_mask()
        pattern = self.get_raw_pattern()
        if self.normalizer:
            pattern = self.normalizer(self.get_raw_pattern())
        x = (1 - mask) * x + mask * pattern
        return self.classifier(x)

    def get_raw_mask(self):
        mask = nn.Tanh()(self.mask_tanh)
        return mask / (2 + self.epsilon) + 0.5

    def get_raw_pattern(self):
        pattern = nn.Tanh()(self.pattern_tanh)
        return pattern / (2 + self.epsilon) + 0.5

    def get_normalizer(self):
        norm = None
        if self.trainer.dataset.name.lower() == 'cifar10':
            # a function to denormalize the image based on cifar10 dataset  [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
            norm = transforms.Normalize(
                [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261])
        elif self.trainer.dataset.name.lower() == 'mnist':
            norm = transforms.Normalize(mean=[0.5], std=[0.5])
        elif self.trainer.dataset.name.lower() == 'imagenet':
            norm = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        return norm

    def get_denormalizer(self):
        denorm = None
        if self.trainer.dataset.name.lower() == 'cifar10':
            # a function to denormalize the image based on cifar10 dataset  [0.4914, 0.4822, 0.4465], [0.247, 0.243, 0.261]
            denorm = transforms.Normalize(
                [-0.4914, -0.4822, -0.4465], [1/0.247, 1/0.243, 1/0.261])
        elif self.trainer.dataset.name.lower() == 'mnist':
            denorm = transforms.Normalize(mean=[-0.5], std=[1/0.5])
        elif self.trainer.dataset.name.lower() == 'imagenet':
            denorm = transforms.Normalize(
                mean=[-0.485, -0.456, -0.406], std=[1/0.229, 1/0.224, 1/0.225])

        return denorm
